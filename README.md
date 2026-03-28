# Lens — AI Image Detector

<img src="icons/icon128.png" alt="Lens icon" width="64">

A browser extension that tells you whether an image was AI-generated, as you browse. Runs entirely in your browser — no images are uploaded, no account needed.

**[Read the full write-up →](https://thevgergroup.com/blog/lens-seeing-through-ai-generated-images/)**

---

## Install

### Firefox
[Download lens-1.1.0.xpi](https://thevgergroup.com/blog/lens-seeing-through-ai-generated-images/) — direct install, no store account needed.

### Chrome, Edge, Brave
Our Chrome Web Store listing is under review. In the meantime:

1. Download and unzip [lens-1.1.0.zip](https://thevgergroup.com/blog/lens-seeing-through-ai-generated-images/) to a permanent folder
2. Go to `chrome://extensions/` and enable **Developer mode**
3. Click **Load unpacked** → select the unzipped folder
4. You can turn Developer mode off again after installing

### Safari
Requires Xcode packaging via `xcrun safari-web-extension-converter`. We're working on an App Store submission.

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

---

## How detection works

Lens runs up to four checks on every image, stopping as soon as it's confident:

1. **URL & filename** — recognises known AI image CDNs (OpenAI, Midjourney, Stability AI, Adobe Firefly, xAI, Bing Image Creator, etc.) and filename patterns
2. **Metadata** — reads EXIF software tags, XMP `CreatorTool`, IPTC `DigitalSourceType`, and embedded [C2PA Content Credentials](https://c2pa.org/)
3. **Pixel statistics** — analyses noise patterns, gradient smoothness, and inter-block noise correlation to catch AI generation artifacts without metadata
4. **Frequency analysis** — detects spread-spectrum watermarks including Google SynthID

For a detailed technical explanation see [WHITEPAPER.md](WHITEPAPER.md).

### Limitations

- **Social media** (Twitter/X, Instagram, Facebook) strips metadata on upload — pixel and frequency checks still run but are weaker
- **FLUX and untagged SDXL** produce pixel statistics that overlap with real photographs — without C2PA or a telltale filename, Lens may not flag them
- Images behind strict CORS policies may not be decodable for pixel analysis

---

## Privacy

- No images or data ever leave your browser
- No account, no telemetry, no analytics
- Image bytes are analysed in memory and immediately discarded
- Only URL hashes are cached for the duration of your browser session

---

## Development

No build step. Plain ES modules that run directly in the browser.

```bash
npm install                      # test dependencies only
npm run fixture:download         # download test images
npm test                         # unit tests (no browser needed)
npm run test:integration         # full integration tests
```

See [CLAUDE.md](CLAUDE.md) for architecture details and how to add new detection signals.

---

## Contributing

Issues and PRs welcome, especially:
- Detection signals for models not currently caught (FLUX, untagged SDXL)
- False positive reports with a URL or image
- New AI model fixture images for the test suite

Please open an issue before a large PR.

---

## License

MIT — see [LICENSE](LICENSE)

Built by [The Vger Group](https://thevgergroup.com)
