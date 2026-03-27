# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LENS** is a browser extension (Manifest V3) that detects AI-generated images on any webpage. It runs entirely client-side with zero external requests or dependencies.

## Development Setup

No build system. No npm. Load directly as an unpacked extension:

- **Chrome/Edge/Brave**: `chrome://extensions/` → Enable Developer mode → Load unpacked → select this folder
- **Firefox**: `about:debugging#/runtime/this-firefox` → Load Temporary Add-on → select `manifest.json`
- **Safari**: `xcrun safari-web-extension-converter ./ai-image-detector --project-location ./safari-ext`

## Debugging

- **Service worker**: `chrome://extensions/` → click "service worker" link under the extension
- **Content script**: DevTools on any page → Sources → Content Scripts
- **Popup**: Right-click the extension popup → Inspect

## Architecture

### Detection Pipeline (`lib/detector.js`)

Four layers run cheapest-first with bail-early on high confidence:

| Layer | Method | Cost | Max Score |
|-------|--------|------|-----------|
| L1 | URL/hostname heuristics | ~0ms | 0.95 |
| L2 | EXIF/XMP/IPTC metadata parsing | ~2ms | 0.97 |
| L3 | Statistical pixel analysis (LSB entropy, gradient smoothness, color correlation) | ~20ms | 0.55 |
| L4 | FFT frequency analysis (SynthID watermark detection) | ~80ms | 0.84 |

Confidence thresholds → badge labels: ≥0.90 Definite, ≥0.70 Likely, ≥0.45 Possible, ≥0.20 Unlikely, <0.20 Clean.

### Service Worker (`background/service-worker.js`)

Central orchestrator. Runs in privileged context to:
- Fetch cross-origin images (bypassing CORS that would block content scripts)
- Decode images via `ImageDecoder` API or `createImageBitmap` fallback on `OffscreenCanvas`
- Cache results: in-memory Map (500-entry LRU) + `chrome.storage.session`
- Limit concurrency to 4 simultaneous analyses

### Content Script (`content/content.js`)

DOM observer and badge injector. Uses:
- `MutationObserver` to catch dynamically added images (SPAs, infinite scroll)
- `IntersectionObserver` to defer analysis until images are near-visible (200px margin)
- `WeakSet` to prevent re-analyzing the same DOM node
- Wraps images in `.lens-wrapper` (position:relative) and injects `.lens-badge`

### Storage

- `chrome.storage.sync` → user settings (persisted across sessions)
- `chrome.storage.session` → URL hash cache (cleared on browser close)
- In-memory `resultCache` Map → fast lookup, max 500 entries

### Message Flow

Content script → `ANALYZE_IMAGE` → Service worker → runs detection → returns result → Content script injects badge → Popup polls SW every 600ms for updated tab stats.

## Testing

### Setup (first time)
```bash
npm install
npm run fixture:download        # downloads C2PA sample images from contentauth
python3 tests/fixtures/download-diffusiondb.py  # optional: 20 SD images for L3 tests
```

### Unit tests (no browser, fast)
```bash
npm test                        # all unit tests
npm run test:unit               # same
npx vitest run tests/unit/detector-l1-url.test.js   # single file
```

Unit tests cover each detection layer in isolation using synthetic `ImageData` and hand-crafted JPEG/PNG buffers. No browser or fixtures required.

### Integration tests (requires agent-browser)
```bash
npm run test:integration        # fixture server + all integration tests
SKIP_LIVE=1 npm run test:all    # unit + integration, skip internet-dependent tests
bash tests/run-all.sh --unit-only
```

Integration tests use `agent-browser --extension .` to load the extension into Chromium and navigate to local fixture pages served by `npx serve`. They assert `.lens-badge` elements appear and check confidence class names.

`agent-browser.json` in the project root configures the extension path and headed mode — it is picked up automatically by `agent-browser`.

### Fixture pages
Served from `tests/fixtures/` at `http://localhost:3456`:
- `pages/ai-images.html` — known AI images (C2PA + DiffusionDB)
- `pages/real-images.html` — known real photographs (C2PA)
- `pages/mixed-page.html` — simulated news feed (mixed AI/real + tiny images to ignore)
- `pages/url-heuristics.html` — tests L1 fast-path with AI-named filenames

### Ground truth image sources
- **C2PA fixtures**: downloaded automatically by `npm run fixture:download` from `contentauth.github.io/example-assets`
- **DiffusionDB (CC0)**: 20 Stable Diffusion images via `download-diffusiondb.py` — good for L3 pixel stats
- **SynthID**: generate via `aitestkitchen.withgoogle.com` (ImageFX) → save to `tests/fixtures/images/ai/synthid-*.png`

### Screenshots
Integration tests save screenshots to `tests/integration/screenshots/` for visual review.

## Key Constraints

- **No external HTTP requests** — all analysis is local
- **No npm/build step** — plain ES modules, runs directly in browser
- Image bytes are never persisted; all analysis is ephemeral
- Social platforms strip EXIF on upload, so L2 is often unavailable for social media images
- `no-cors` opaque responses mean pixel analysis (L3/L4) may be unavailable for some cross-origin images — degrade gracefully
