# Lens — Technical Whitepaper

**Client-side AI image detection in a browser extension**

*The Vger Group — March 2026*

Full write-up: [thevgergroup.com/blog/lens-seeing-through-ai-generated-images](https://thevgergroup.com/blog/lens-seeing-through-ai-generated-images/)

---

## Overview

Lens is a browser extension that detects AI-generated images without sending any data to a server. All analysis runs inside the browser's service worker using standard Web APIs. This document describes the detection architecture, the rationale for each layer, and the known limitations.

---

## Architecture

### Design constraints

- **No external requests.** Every byte of analysis runs locally. No image is ever uploaded.
- **No build step.** The extension is plain ES modules loaded directly by the browser — no bundler, no transpilation.
- **No npm dependencies at runtime.** The only `devDependencies` are test tooling.
- **Graceful degradation.** Each layer is independent. If pixel decoding fails (e.g. opaque cross-origin response), the extension falls back to the layers that still work.

### Execution model

The content script (`content/content.js`) observes the DOM and dispatches `ANALYZE_IMAGE` messages to the service worker (`background/service-worker.js`) for each image that enters the viewport. The service worker fetches the image, decodes it, runs the detection pipeline, and returns a result. The content script then injects a badge overlay.

Concurrency is capped at 4 simultaneous analyses. Results are cached in an in-memory LRU map (500 entries) and in `chrome.storage.session` for the duration of the browser session.

---

## Detection pipeline

The pipeline runs cheapest-first and bails out as soon as the accumulated confidence crosses 0.90. In practice this means URL and metadata checks handle the majority of detected images at near-zero cost, and the heavier pixel and frequency layers only run when those checks are inconclusive.

### Layer 1 — URL and filename heuristics (~0ms)

Many AI images retain their origin trail. A URL containing `oaidalleapiprodscus.blob.core.windows.net` identifies a DALL-E image with near certainty before any pixels are examined. Lens maintains a set of known AI CDN hostnames (OpenAI, Midjourney, Adobe Firefly, Stability AI, Leonardo, xAI Aurora, Ideogram, NightCafe, and others) and a set of path/filename patterns.

Signal weights:
- Known AI CDN hostname: **0.95**
- AI URL path pattern: **0.80**
- AI filename pattern (e.g. `DALL-E`, `MJ-`): **0.60**

### Layer 2 — Embedded metadata (~2ms)

Modern AI tools embed machine-readable provenance in several places. Lens parses these without any external library, using a hand-written JPEG marker walker and PNG chunk reader:

**EXIF (APP1 / IFD0)**
Tag `0x0131` (Software) is compared against a list of known AI tool strings. ChatGPT images typically contain `gpt-image-1`; Adobe Firefly contains `Adobe Firefly`; Midjourney images exported from the web app contain `Midjourney`. Weight: **0.92**.

**XMP (also in APP1)**
The `xmp:CreatorTool` field is checked against the same list. Lens also checks for the Adobe Firefly-specific `GenerativeAI` marker and the `firefly` namespace. Weight: **0.92–0.95**.

**IPTC (APP13)**
The IPTC Digital Source Type controlled vocabulary defines `trainedAlgorithmicMedia` specifically for AI-generated content. When present this is the strongest single signal. Weight: **0.97**.

**C2PA Content Credentials (APP11 / JUMBF)**
C2PA (Coalition for Content Provenance and Authenticity) is an emerging standard for embedding signed provenance chains in images. ChatGPT, Adobe Firefly, and Bing Image Creator all produce C2PA-credentialled images. Lens detects C2PA in two ways:

1. APP11 JUMBF binary blocks: scans for the `c2pa.` label prefix in the binary content. Weight: **0.95**.
2. XMP namespace declaration: requires `xmlns:c2pa=` or `<c2pa:` prefix (not a bare substring match, which would falsely match provenance URLs). Weight: **0.88**.

> **False positive note:** The JUMBF block in real photographs can contain `c2pa.ai_generative_training` — an assertion about training rights, not about generation. Lens excludes `generative` as a standalone match term and requires the full `c2pa.` assertion prefix with known generation-claim labels.

### Layer 3 — Statistical pixel analysis (~20ms)

When metadata has been stripped — which happens routinely on social media platforms — Lens analyses the raw pixel data decoded via `OffscreenCanvas` / `ImageDecoder`.

Three statistical signals are computed:

**LSB entropy**
The least significant bit of each colour channel is extracted across a random sample of pixels. Real photographs, which carry camera sensor noise, produce LSB distributions close to 50% for each channel. AI-generated images processed through lossy codecs tend to show biased LSB distributions. A per-channel entropy score is computed; entropy below 0.92 bits (out of 1.0) is flagged as a weak signal. Weight: **0.35–0.45** depending on severity.

**Gradient smoothness**
AI-generated images tend to have smoother gradients than photographs of the same apparent complexity, especially in background regions. Lens computes the mean absolute difference between adjacent pixels across a grid sample. Very low gradient variance relative to image mean luminance is a weak positive signal. Weight: **0.35**.

**Inter-block noise correlation (Mallet et al. 2025)**
This is the most discriminative pixel-domain signal. The algorithm:

1. Apply a Laplacian filter to isolate the noise residual (subtracting low-frequency content)
2. Divide the image into 8×8 pixel blocks
3. Select the 30 blocks with the lowest noise variance (i.e. the "flattest" regions)
4. Compute pairwise Pearson correlation between all selected block noise vectors
5. The fraction of pairs with |r| > 0.3 is the `highCorrFrac` summary statistic

In real photographs, the noise residual in flat regions is dominated by camera sensor noise, which is spatially uncorrelated between different image locations. AI-generated images produced by diffusion models tend to show structured noise that correlates across distant blocks — an artefact of the denoising process.

**Browser calibration note:** JPEG/WebP decode in `OffscreenCanvas` applies slightly different dithering and chroma upsampling than server-side decoders (Node.js / libvips / ImageMagick). Real photographs decoded in-browser score `hcf` in the range 0.18–0.46; AI-generated images (Aurora specifically) score 0.50–0.70. The threshold is set at **hcf > 0.500** based on empirical calibration across 24 real and 12 AI images. Weight: **0.45–0.52**.

**Detectable models (L3):** xAI Aurora (reliable). FLUX.1 and SDXL without metadata are not reliably distinguishable from photographs at this threshold — their statistical properties overlap too much.

### Layer 4 — Frequency domain analysis (~80ms)

Some AI systems embed invisible watermarks using spread-spectrum techniques. Lens computes the 2D FFT of the luminance channel and analyses the frequency magnitude spectrum for structured peaks inconsistent with natural image statistics.

Natural images follow a 1/f power spectrum (more energy at low frequencies, falling off smoothly). Spread-spectrum watermarks create periodic peaks in the mid-frequency range. Lens looks for:

- Periodic spikes in the FFT magnitude spectrum at non-DC frequencies
- Harmonic relationships between peaks (consistent with a carrier signal)
- Spatial symmetry in the spectral peaks (consistent with SynthID's reported architecture)

This layer is specifically intended to detect Google SynthID watermarks, but may also catch other spread-spectrum approaches. Weight: **0.75–0.84**.

---

## Confidence scoring and badge levels

Signals from all layers are accumulated. The final score is the maximum of all signal weights (not a sum), with a small additive bonus when multiple independent signals agree.

| Score | Badge | Label |
|-------|-------|-------|
| ≥ 0.90 | 🔴 | AI Generated |
| ≥ 0.70 | 🟠 | Likely AI |
| ≥ 0.45 | 🟡 | Possible AI |
| ≥ 0.20 | 🟢 | Probably Real |
| < 0.20 | 🔵 | No AI signals |

The default display threshold is 0.45 (Possible AI and above). Users can lower or raise this in the popup.

---

## Known limitations

**Social media metadata stripping.** Twitter/X, Instagram, and Facebook remove EXIF, XMP, IPTC, and C2PA data on upload. For images shared on these platforms, only L3 and L4 are available — and L3/L4 are weaker and slower.

**FLUX.1 and untagged SDXL.** These models produce images with pixel statistics that overlap significantly with real photographs in browser-decoded colour space. Without a metadata tag or a recognisable CDN URL, Lens will not reliably flag them. This is an active research area.

**CORS and opaque responses.** When a page embeds a cross-origin image with restrictive CORS headers and no CORS response headers, the service worker receives an opaque response with zeroed pixel data. L3 and L4 are unavailable; L1 and L2 still run.

**Adversarial post-processing.** An image that has been JPEG-recompressed multiple times, resized, or had its metadata deliberately stripped will score lower. Lens is designed for the common case, not adversarial stripping.

**It is not a fact-checker.** Lens detects AI generation signals — it does not assess whether an image is misleading, out of context, or manipulated in other ways.

---

## Privacy design

The privacy model is non-negotiable: no data leaves the browser.

- Images are fetched by the service worker and decoded into `ImageData` in memory
- Pixel analysis is performed on the raw bytes; the buffer is immediately discarded after analysis
- Only a SHA-256 hash of the image URL is stored in `chrome.storage.session` to avoid re-analysing the same image
- Session storage is cleared when the browser closes
- No telemetry, no analytics, no crash reporting, no external API calls of any kind

---

## References

- C2PA specification: [c2pa.org](https://c2pa.org/)
- IPTC Photo Metadata Standard — Digital Source Type: [iptc.org](https://iptc.org/standards/photo-metadata/)
- Mallet et al. (2025), *Forensic Detection of AI-Generated Images via Block Noise Correlation*
- Fernandez et al. (2023), *The Stable Signature: Rooting Watermarks in Latent Diffusion Models*
- Google SynthID: [deepmind.google/synthid](https://deepmind.google/technologies/synthid/)

---

*Lens is open source — MIT licence. Source: [github.com/thevgergroup/lens](https://github.com/thevgergroup/lens)*
*Built by [The Vger Group](https://thevgergroup.com)*
