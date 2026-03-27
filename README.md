# Lens — AI Image Detector

A browser extension that analyses images on any web page for AI generation signals, running entirely client-side with no data leaving the browser.

## Detection Layers

Analysis runs cheapest-first, bailing out early when confidence is sufficient:

| Layer | Method | Speed | Notes |
|-------|--------|-------|-------|
| L1 | URL/hostname heuristics | ~0ms | Known AI CDN domains, filename patterns |
| L2 | EXIF/XMP/IPTC/PNG metadata | ~2ms | Software tags, C2PA, IPTC `trainedAlgorithmicMedia` |
| L3 | Statistical pixel analysis | ~20ms | LSB entropy, gradient smoothness, noise floor |
| L4 | FFT frequency analysis | ~80ms | SynthID-inspired spread-spectrum phase detection |

## Browser Support

| Browser | Engine | MV3 | Min Version |
|---------|--------|-----|-------------|
| Chrome  | Chromium | ✅ Full | 109+ |
| Edge    | Chromium | ✅ Full | 109+ |
| Firefox | Gecko   | ✅ Full | 109+ |
| Brave   | Chromium | ✅ Full | Latest |
| Opera   | Chromium | ✅ Full | Latest |
| Safari  | WebKit  | ⚠️ Requires Xcode packaging | 16+ |

## Installation (Developer Mode)

### Chrome / Edge / Brave / Opera

1. Clone or download this repository
2. Go to `chrome://extensions` (or `edge://extensions`)
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked**
5. Select the `ai-image-detector` folder

### Firefox

1. Go to `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on**
3. Select the `manifest.json` file inside `ai-image-detector`

Note: For permanent Firefox installation, the extension needs to be signed via AMO or use `about:config` to set `xpinstall.signatures.required = false`.

### Safari

Safari requires Xcode to convert the extension:

```bash
xcrun safari-web-extension-converter ./ai-image-detector --project-location ./safari-ext
```

Then open the generated Xcode project, build, and enable in Safari → Preferences → Extensions.

## Project Structure

```
ai-image-detector/
├── manifest.json              # MV3 manifest (Chrome + Firefox + Safari)
├── background/
│   └── service-worker.js      # Orchestrates analysis, manages cache, fetches images
├── content/
│   ├── content.js             # DOM observation, badge injection
│   └── content.css            # Badge styles
├── popup/
│   ├── popup.html             # Extension popup
│   ├── popup.js               # Popup logic
│   └── popup.css              # Popup styles (forensic terminal aesthetic)
├── lib/
│   └── detector.js            # Core detection engine (all 4 layers)
└── icons/
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

## Architecture Notes

**Why a Service Worker for analysis?**

- Avoids blocking the page's main thread
- Has access to `OffscreenCanvas` for pixel manipulation
- Can `fetch()` cross-origin images without CORS restrictions (the SW acts as a privileged proxy)
- Shared across all tabs — analysis cache persists for the browser session

**CORS & Image Access**

Normal page scripts are blocked from reading cross-origin image pixels into canvas (CORS). The service worker bypasses this by fetching images directly with the extension's host permissions, which are granted for `<all_urls>`.

**Result Caching**

Results are cached by URL (in-memory Map, up to 500 entries). The same image appearing on multiple pages or in multiple places on the same page is only analysed once per session.

**MutationObserver + IntersectionObserver**

New images added dynamically (SPAs, infinite scroll, lazy loading) are picked up by a `MutationObserver`. Analysis is deferred until images enter the viewport via `IntersectionObserver` to avoid wasting CPU on off-screen content.

## Adding L5: ML Model Detection

To add an on-device ML detector (e.g. via ONNX Runtime Web or Transformers.js):

```javascript
// In service-worker.js
import * as ort from 'onnxruntime-web/webgpu';

let session;
async function loadModel() {
  if (session) return session;
  session = await ort.InferenceSession.create(
    chrome.runtime.getURL('models/ai_detector.onnx'),
    { executionProviders: ['webgpu', 'wasm'] }
  );
  return session;
}
```

Recommended models (from Hugging Face):
- `umm-maybe/AI-image-detector` (~45MB)
- `haywoodsloan/ai-image-detector-deploy` (~90MB)

Both are binary classifiers (real vs AI-generated) that can be exported to ONNX format.

## Known Limitations

| Limitation | Impact |
|------------|--------|
| Social platforms (Twitter/X, Facebook, LinkedIn) strip all EXIF metadata on upload | Layer 2 mostly unavailable |
| SynthID detection is based on reverse-engineering of 250 Gemini images; may not generalise perfectly | Layer 4 is heuristic, not definitive |
| Google can rotate SynthID carrier frequencies at any time | Layer 4 may become outdated |
| ML models have ~10-15% false positive rates | Badge design uses confidence %, not binary labels |
| `no-cors` fetch returns opaque responses — pixel analysis may fail for some cross-origin images | Graceful degradation to URL/metadata layers |

## Privacy

- Zero network requests to external servers
- Images are decoded in the extension service worker and immediately discarded
- Only URL hashes are stored in `chrome.storage.session` (not image data)
- All analysis is ephemeral — cleared when the browser session ends

## License

MIT
