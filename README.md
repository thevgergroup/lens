# Lens — AI Image Detector

A browser extension that tells you whether an image was AI-generated, as you browse. Runs entirely in your browser — no images are uploaded, no account needed.

---

## What you see

Lens overlays a small badge on every image it analyses. Hover the badge to see why.

| Badge | Meaning | Confidence |
|-------|---------|------------|
| 🔴 **AI Generated** | Strong evidence — verified C2PA metadata, known AI tool signature, or frequency watermark | ≥ 90% |
| 🟠 **Likely AI** | Multiple corroborating signals | ≥ 70% |
| 🟡 **Possible AI** | One or more weak signals | ≥ 45% |
| 🟢 **Probably Real** | Very weak signals, likely a false alarm | ≥ 20% |
| 🔵 **No AI signals** | Nothing detected | < 20% |

By default only the top three levels show badges. You can change the threshold in the popup.

---

## How detection works

Lens runs up to four checks on every image, stopping as soon as it's confident:

1. **URL & filename** — recognises known AI image CDNs (OpenAI, Midjourney, Stability AI, Adobe, etc.) and filename patterns like `dalle-`, `flux_`, `midjourney`
2. **Metadata** — reads EXIF software tags, XMP `CreatorTool`, IPTC `DigitalSourceType`, and embedded [C2PA Content Credentials](https://c2pa.org/) from tools like ChatGPT, Adobe Firefly, and Bing Image Creator
3. **Pixel statistics** — analyses noise patterns, gradient smoothness, and inter-block noise correlation to catch AI generation artifacts without metadata
4. **Frequency analysis** — detects spread-spectrum watermarks such as Google SynthID

**Works on:** ChatGPT / DALL-E, Adobe Firefly, Midjourney, Bing Image Creator, Stable Diffusion, SDXL, FLUX, xAI Aurora, and more.

### Honest limitations

- **Social media** (Twitter/X, Instagram, Facebook) strips metadata on upload. Pixel and frequency checks still run but are weaker.
- **FLUX and SDXL without metadata** are hard to detect — their pixel statistics overlap with real photographs. If there's no C2PA tag or telltale filename, Lens may not catch them.
- Images behind strict CORS policies may not be decodable for pixel analysis — Lens falls back to URL and metadata checks.

---

## Privacy

- No images or data ever leave your browser
- No account, no telemetry, no analytics
- Image bytes are analysed in memory and immediately discarded — nothing is written to disk
- Only URL hashes are cached for the duration of your browser session

---

## Install

### Chrome, Edge, Brave, Opera

1. Download or clone this repository
2. Go to `chrome://extensions/`
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked** → select the repository folder

### Firefox

1. Go to `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on** → select `manifest.json`

> For a permanent Firefox install the extension needs to be signed. See [Firefox Extension Workshop](https://extensionworkshop.com/documentation/publish/).

### Safari

```bash
xcrun safari-web-extension-converter . --project-location ./safari-ext
```

Open the generated Xcode project, build, then enable in Safari → Settings → Extensions.

---

## Browser support

| Browser | Support |
|---------|---------|
| Chrome 109+ | ✅ Full |
| Edge 109+ | ✅ Full |
| Firefox 109+ | ✅ Full |
| Brave / Opera | ✅ Full |
| Safari 16+ | ⚠️ Requires Xcode packaging |

---

## Development

No build step. Plain ES modules that run directly in the browser.

```bash
npm install                      # test dependencies only
npm run fixture:download         # download test images
npm test                         # unit tests (no browser needed)
npm run test:integration         # full integration tests
npm run fixture:serve            # serve test pages at localhost:3456
```

See [CLAUDE.md](CLAUDE.md) for architecture details, how the detection pipeline works, and how to add new signals.

---

## Contributing

Issues and PRs welcome, especially:
- New AI model fixture images we don't yet test
- Detection signals for models not currently caught
- False positive reports with a URL or image

Please open an issue before a large PR.

---

## License

MIT
