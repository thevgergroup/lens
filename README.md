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

## ML Model Training

The L6 forensic NPR encoder is trained with [DVC](https://dvc.org), which tracks data versions, hyperparameters, and metrics reproducibly. Training images live in S3 (`s3://lens-training-data/dvc`) and are never committed to git. Model outputs (`lib/forensic-ai-detector.keras`, `lib/forensic-tfjs/`) are DVC-cached.

### First-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
dvc pull                         # fetch all training data from S3 (~2GB)
```

### Run training

```bash
dvc repro train-forensic         # train; skips if deps/params unchanged
dvc metrics show                 # print AUC, precision, recall, FPR
```

DVC checks whether any dependency (training script, image dirs, `params.yaml`) has changed since the last cached run. If nothing changed the stage is skipped. Use `--force` to retrain unconditionally.

### Hyperparameter experiments

```bash
dvc exp run                                  # run with current params.yaml
dvc exp run -S train.epochs=20               # override a single param inline
dvc exp run -S train.lr=0.0005 -S train.epochs=25
dvc exp show                                 # compare all runs side-by-side
dvc exp diff                                 # diff last two experiments
```

### WandB run naming

Runs are named `forensic-npr_{data-hash}_{epochs}e_{lr}lr_{seed}s` — e.g. `forensic-npr_55ff2159_15e_1e-3lr_42s`. The `data-hash` is the first 8 chars of the combined MD5 of all `.dvc` pointer files, so every wandb run and DVC experiment is traceable to an exact dataset snapshot.

### Adding new training data

```bash
# Download new images to the right dir, then:
dvc add tests/fixtures/images/ai/<source>
dvc push
dvc repro train-forensic         # DVC detects the new dep and retrains
```

### Training data sources

| Source | Class | Count | Script |
|--------|-------|-------|--------|
| Defactify (SD 2.1 / SD3 / SDXL) | AI | 1200 | `download-defactify.py` |
| DALL-E 3 (HuggingFace) | AI | 400 | `download-dalle3.py` |
| Midjourney (HuggingFace) | AI | 400 | `download-midjourney.py` |
| Grok / Aurora (X CDN) | AI | 48 | `download-grok.py` |
| Kaggle (misc generators) | AI | 50 | `download-training-data.py` |
| MS COCO | Real | 1000 | `download-training-data.py` |
| SUN397 | Real | 1000 | `download-training-data.py` |
| Kaggle (real photos) | Real | 745 | `download-training-data.py` |
| Picsum + Wikimedia (curated) | Real | 84 | `download-real-photos.py` |

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
