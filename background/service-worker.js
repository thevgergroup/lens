/**
 * LENS — Service Worker
 * Handles image fetching (bypasses CORS), orchestrates detection pipeline,
 * and caches results to avoid re-analysis.
 */

import {
  checkUrlHeuristics,
  parseMetadata,
  analyzePixelStatistics,
  analyzeFrequencyDomain,
  interpretScore,
  CONFIDENCE_THRESHOLDS,
} from '../lib/detector.js';

import {
  loadGraphModel,
  tensor4d,
  tidy,
  setBackend,
  ready,
} from '../lib/tfjs/tfjs.min.js';

// ---------------------------------------------------------------------------
// MobileNetV3 model — loaded once, reused for all analyses
// ---------------------------------------------------------------------------

// ── Model loading ─────────────────────────────────────────────────────────────

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

let mobileNetModel = null;
let mobileNetLoading = null;
let forensicModel = null;
let forensicLoading = null;
let tfBackendReady = null;

async function ensureTFBackend() {
  if (tfBackendReady) return tfBackendReady;
  tfBackendReady = (async () => {
    // MV3 service workers lack URL.createObjectURL, so WASM backend is unavailable.
    // CPU backend (pure JS) works in SW context without DOM APIs.
    await setBackend('cpu');
    await ready();
  })();
  return tfBackendReady;
}

async function getMobileNet() {
  if (mobileNetModel) return mobileNetModel;
  if (mobileNetLoading) return mobileNetLoading;
  mobileNetLoading = (async () => {
    try {
      await ensureTFBackend();
      const modelUrl = chrome.runtime.getURL('lib/mobilenet-tfjs/model.json');
      mobileNetModel = await loadGraphModel(modelUrl);
      console.log('[Lens SW] MobileNet model loaded');
    } catch (err) {
      console.warn('[Lens SW] MobileNet load failed:', err.message);
      mobileNetModel = null;
    }
    return mobileNetModel;
  })();
  return mobileNetLoading;
}

async function getForensicModel() {
  if (forensicModel) return forensicModel;
  if (forensicLoading) return forensicLoading;
  forensicLoading = (async () => {
    try {
      await ensureTFBackend();
      const modelUrl = chrome.runtime.getURL('lib/forensic-tfjs/model.json');
      forensicModel = await loadGraphModel(modelUrl);
      console.log('[Lens SW] Forensic model loaded');
    } catch (err) {
      console.warn('[Lens SW] Forensic model not available:', err.message);
      forensicModel = null;
    }
    return forensicModel;
  })();
  return forensicLoading;
}

// Kick off model loading on SW startup
getMobileNet();
getForensicModel();

// ── Preprocessing ─────────────────────────────────────────────────────────────

// NPR (Neighboring Pixel Residual) transform.
// Downsample 2x nearest-neighbor then upsample 2x, subtract from original.
// Isolates upsampling artifacts common to all generative models (GANs, diffusion).
// Content is suppressed; forensic artifacts are amplified.
// Mirrors the Python apply_npr() in train-forensic-encoder.py exactly.
function applyNPR(rgba, width, height, outSize) {
  const buf = new Float32Array(outSize * outSize * 3);
  const xScale = width  / outSize;
  const yScale = height / outSize;

  for (let y = 0; y < outSize; y++) {
    for (let x = 0; x < outSize; x++) {
      // Bilinear sample from source
      const srcX = x * xScale, srcY = y * yScale;
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, width - 1);
      const y1 = Math.min(y0 + 1, height - 1);
      const dx = srcX - x0, dy = srcY - y0;

      // Nearest-neighbor 2x downsample+upsample: snap to even pixel
      const nx = (x0 & ~1), ny = (y0 & ~1);
      const nx1 = Math.min(nx, width - 1), ny1 = Math.min(ny, height - 1);

      const base = (y * outSize + x) * 3;
      for (let c = 0; c < 3; c++) {
        const i00 = (y0 * width + x0) * 4 + c;
        const i10 = (y0 * width + x1) * 4 + c;
        const i01 = (y1 * width + x0) * 4 + c;
        const i11 = (y1 * width + x1) * 4 + c;
        const pixel = (rgba[i00]*(1-dx)*(1-dy) + rgba[i10]*dx*(1-dy) +
                       rgba[i01]*(1-dx)*dy      + rgba[i11]*dx*dy) / 255;

        const ni = (ny1 * width + nx1) * 4 + c;
        const recon = rgba[ni] / 255;

        buf[base + c] = (pixel - recon) * (2 / 3);
      }
    }
  }
  return buf;
}

function preprocessImageData(imageData) {
  const { width, height, data } = imageData;
  const SIZE = 224;
  const buf = new Float32Array(SIZE * SIZE * 3);
  const xScale = width  / SIZE;
  const yScale = height / SIZE;

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const srcX = x * xScale;
      const srcY = y * yScale;
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, width  - 1);
      const y1 = Math.min(y0 + 1, height - 1);
      const dx = srcX - x0, dy = srcY - y0;

      const idx = (c) => {
        const i00 = (y0 * width + x0) * 4 + c;
        const i10 = (y0 * width + x1) * 4 + c;
        const i01 = (y1 * width + x0) * 4 + c;
        const i11 = (y1 * width + x1) * 4 + c;
        return (data[i00] * (1-dx)*(1-dy) + data[i10] * dx*(1-dy) +
                data[i01] * (1-dx)*dy     + data[i11] * dx*dy) / 255;
      };

      const base = (y * SIZE + x) * 3;
      buf[base]     = (idx(0) - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      buf[base + 1] = (idx(1) - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      buf[base + 2] = (idx(2) - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  return buf;
}

// In-memory result cache: URL hash → result
// Survives across content script navigations within a browser session
const resultCache = new Map();
const MAX_CACHE_SIZE = 500;

// Debug log ring buffer — last 200 results, readable by popup
const debugLog = [];
const MAX_DEBUG_LOG = 200;

// Feature flags — toggled via settings, persisted in chrome.storage.sync
// disableL5: true by default — MobileNet v1.2 has calibration issues causing
// high FPR on real photos. Enable experimentally via Settings → Debug tab.
let featureFlags = { disableL5: true };
chrome.storage.sync.get('lensFlags', d => { if (d.lensFlags) featureFlags = { ...featureFlags, ...d.lensFlags }; });

// Per-tab URL tracking for badge counts: tabId → Set<url>
const tabUrls = new Map();

// Concurrent analysis throttle — don't hammer all images at once
const analysisQueue = new Map(); // url → Promise
const MAX_CONCURRENT = 4;
let activeCount = 0;

// Global SW keepalive — prevents suspension between sequential analyses.
// Fires every 10s; cleared 5s after the last analysis completes.
let globalKeepalive = null;
let keepaliveExpiry = null;

function touchKeepalive() {
  keepaliveExpiry = Date.now() + 5000;
  if (globalKeepalive) return;
  globalKeepalive = setInterval(() => {
    if (Date.now() > keepaliveExpiry) {
      clearInterval(globalKeepalive);
      globalKeepalive = null;
    } else {
      chrome.runtime.getPlatformInfo(() => {});
    }
  }, 10000);
}

// ---------------------------------------------------------------------------
// Message handler (from content scripts)
// ---------------------------------------------------------------------------

self.addEventListener('message', async (event) => {
  const { type, payload, id } = event.data;

  switch (type) {
    case 'ANALYZE_IMAGE':
      // Legacy postMessage path — tab ID not available, no-op here
      break;

    case 'GET_CACHED':
      event.source.postMessage({
        type: 'CACHE_RESULT',
        id,
        result: resultCache.get(payload.url) || null,
      });
      break;

    case 'GET_PAGE_STATS':
      handleGetPageStats(event.source, payload.imageUrls, id);
      break;

    case 'CLEAR_CACHE':
      resultCache.clear();
      chrome.storage.session?.clear().catch(() => {});
      event.source.postMessage({ type: 'CACHE_CLEARED', id });
      break;
  }
});

// Also handle messages from extension popup via chrome.runtime
self.addEventListener('connect', () => {});

// Support runtime.onMessage for content scripts ↔ service worker and popup ↔ service worker
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYZE_IMAGE') {
    const { url, domSignals } = message.payload;
    const shortUrl = url.split('/').pop().slice(0, 40);
    console.log(`[Lens SW] ANALYZE_IMAGE received: ${shortUrl}`);

    const tabId = sender?.tab?.id;

    // Keep SW alive during and between sequential analyses
    touchKeepalive();

    // Check cache for immediate inline response
    if (resultCache.has(url)) {
      console.log(`[Lens SW] Cache hit: ${shortUrl}`);
      const cached = resultCache.get(url);
      if (tabId != null) { trackTabUrl(tabId, url); updateTabBadge(tabId); }
      sendResponse({ result: cached, fromCache: true });
      return false;
    }

    // Per-analysis keepalive as belt-and-suspenders during async fetch+decode
    const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(() => {}), 10000);

    analyzeImage(url, domSignals || {})
      .then(result => {
        cacheResult(url, result);
        logResult(shortUrl, result);
        if (tabId != null) {
          trackTabUrl(tabId, url);
          updateTabBadge(tabId);
          chrome.tabs.sendMessage(tabId, { type: 'ANALYSIS_RESULT', url, result }).catch(() => {});
        }
        touchKeepalive(); // reset idle window before clearing per-analysis keepalive
        try { sendResponse({ result, fromCache: false }); } catch (_) {}
      })
      .catch(err => {
        console.warn(`[Lens SW] Analysis error for ${shortUrl}:`, err.message);
        try { sendResponse({ error: err.message }); } catch (_) {}
      })
      .finally(() => clearInterval(keepAlive));

    return true; // Keep message channel open for async response
  }

  if (message.type === 'GET_TAB_STATS') {
    const urls = message.urls || [];
    const stats = buildStats(urls);
    // Also return the full results map so the popup can render without a separate round-trip
    const resultMap = {};
    for (const url of urls) {
      if (resultCache.has(url)) resultMap[url] = resultCache.get(url);
    }
    sendResponse({ stats, results: resultMap });
    return false;
  }

  if (message.type === 'ANALYZE_IMAGE_POPUP') {
    analyzeImage(message.url, {})
      .then(result => sendResponse({ result }))
      .catch(err => sendResponse({ error: err.message }));
    return true; // Keep channel open for async response
  }

  if (message.type === 'GET_CACHE_SIZE') {
    sendResponse({ size: resultCache.size });
    return false;
  }

  if (message.type === 'CLEAR_CACHE') {
    resultCache.clear();
    chrome.storage.session?.clear().catch(() => {});
    debugLog.length = 0;
    chrome.tabs.query({}, tabs => {
      for (const tab of tabs) {
        chrome.tabs.sendMessage(tab.id, { type: 'REANALYZE' }).catch(() => {});
      }
    });
    sendResponse({ ok: true });
    return false;
  }

  if (message.type === 'GET_DEBUG_LOG') {
    sendResponse({ log: debugLog.slice() });
    return false;
  }

  if (message.type === 'SET_FLAGS') {
    featureFlags = { ...featureFlags, ...message.flags };
    chrome.storage.sync.set({ lensFlags: featureFlags });
    sendResponse({ ok: true, flags: featureFlags });
    return false;
  }
});

// ---------------------------------------------------------------------------
// Core analysis pipeline
// ---------------------------------------------------------------------------

function replyToTab(tabId, message) {
  if (tabId != null) {
    chrome.tabs.sendMessage(tabId, message).catch(() => {});
  }
}

async function handleAnalyzeImage(tabId, url, domSignals, requestId) {
  // Check cache first
  if (resultCache.has(url)) {
    replyToTab(tabId, {
      type: 'ANALYSIS_RESULT',
      id: requestId,
      url,
      result: resultCache.get(url),
      fromCache: true,
    });
    return;
  }

  // Deduplicate concurrent requests for same URL
  if (analysisQueue.has(url)) {
    const result = await analysisQueue.get(url);
    replyToTab(tabId, { type: 'ANALYSIS_RESULT', id: requestId, url, result, fromCache: false });
    return;
  }

  const analysisPromise = analyzeImage(url, domSignals || {});
  analysisQueue.set(url, analysisPromise);

  try {
    const result = await analysisPromise;
    cacheResult(url, result);
    replyToTab(tabId, { type: 'ANALYSIS_RESULT', id: requestId, url, result, fromCache: false });
  } catch (err) {
    replyToTab(tabId, { type: 'ANALYSIS_ERROR', id: requestId, url, error: err.message });
  } finally {
    analysisQueue.delete(url);
  }
}

async function analyzeImage(url, domSignals) {
  const allSignals = [];
  let maxScore = 0;
  const timing = {};
  let layer = 0;

  // ── LAYER 1: URL heuristics (free) ─────────────────────────────────────
  const t1 = Date.now();
  const urlResult = checkUrlHeuristics(url);
  timing.l1 = Date.now() - t1;
  allSignals.push(...urlResult.signals);
  maxScore = Math.max(maxScore, urlResult.score);
  layer = 1;

  // Bail early if already definite
  if (maxScore >= CONFIDENCE_THRESHOLDS.DEFINITE) {
    return buildResult(url, maxScore, allSignals, layer, timing);
  }

  // Incorporate DOM signals from content script
  if (domSignals.altText) {
    const altLower = domSignals.altText.toLowerCase();
    const aiKeywords = ['ai generated', 'ai-generated', 'generated by', 'created by ai',
                        'artificial intelligence', 'stable diffusion', 'midjourney',
                        'dall-e', 'firefly', 'image creator'];
    for (const kw of aiKeywords) {
      if (altLower.includes(kw)) {
        allSignals.push({ type: 'dom', label: `Alt text: "${domSignals.altText.slice(0, 60)}"`, weight: 0.75 });
        maxScore = Math.max(maxScore, 0.75);
        break;
      }
    }
  }

  // ── Fetch image bytes ───────────────────────────────────────────────────
  let arrayBuffer;
  let imageData;

  try {
    const tFetch = Date.now();
    console.log(`[Lens SW] Fetching: ${url.split('/').pop().slice(0, 40)}`);
    arrayBuffer = await fetchImageBytes(url);
    timing.fetch = Date.now() - tFetch;
    console.log(`[Lens SW] Fetched ${arrayBuffer.byteLength} bytes in ${timing.fetch}ms`);
  } catch (err) {
    console.warn(`[Lens SW] Fetch failed:`, err.message);
    return buildResult(url, maxScore, allSignals, layer, timing, 'fetch_failed');
  }

  // ── LAYER 2: Metadata ───────────────────────────────────────────────────
  const t2 = Date.now();
  const metaResult = await parseMetadata(arrayBuffer);
  timing.l2 = Date.now() - t2;
  allSignals.push(...metaResult.signals);
  maxScore = Math.max(maxScore, metaResult.score);
  layer = 2;

  if (maxScore >= CONFIDENCE_THRESHOLDS.DEFINITE) {
    return buildResult(url, maxScore, allSignals, layer, timing);
  }

  // ── Decode to ImageData for pixel analysis ──────────────────────────────
  try {
    const tDecode = Date.now();
    imageData = await decodeImageToData(arrayBuffer);
    timing.decode = Date.now() - tDecode;
  } catch (err) {
    return buildResult(url, maxScore, allSignals, layer, timing, 'decode_failed');
  }

  if (!imageData || imageData.width < 32 || imageData.height < 32) {
    return buildResult(url, maxScore, allSignals, layer, timing, 'too_small');
  }

  // ── LAYER 3: Pixel statistics ───────────────────────────────────────────
  const t3 = Date.now();
  // JPEG/WebP lossy encoding destroys LSB randomness via DCT quantization —
  // skip LSB entropy check on lossy-compressed images to avoid false positives.
  const mime = sniffMimeType(arrayBuffer);
  const isLossy = mime === 'image/jpeg' || mime === 'image/webp';
  const pixelResult = analyzePixelStatistics(imageData, isLossy);
  timing.l3 = Date.now() - t3;
  allSignals.push(...pixelResult.signals);
  maxScore = Math.max(maxScore, pixelResult.score);
  layer = 3;

  // ── LAYER 4: FFT frequency analysis ────────────────────────────────────
  // Only run on larger images where it's meaningful
  if (imageData.width >= 64 && imageData.height >= 64) {
    const t4 = Date.now();
    const fftResult = analyzeFrequencyDomain(imageData);
    timing.l4 = Date.now() - t4;
    allSignals.push(...fftResult.signals);
    maxScore = Math.max(maxScore, fftResult.score);
    layer = 4;
  }

  // ── LAYER 5: MobileNetV3 neural classifier ──────────────────────────────
  // L5 MobileNet: disabled by default — MobileNet v1.2 has calibration issues
  // that cause high false-positive rates on real photos (Picsum, stock photos).
  // Enable via featureFlags.disableL5 = false in settings for experimental use.
  if (!featureFlags.disableL5 && imageData.width >= 64 && imageData.height >= 64) {
    const t5 = Date.now();
    try {
      const model = await getMobileNet();
      if (model) {
        const prob = tidy(() => {
          const pixels = preprocessImageData(imageData);
          const input = tensor4d(pixels, [1, 224, 224, 3]);
          const output = model.predict(input);
          return output.dataSync()[0];
        });
        timing.l5 = Date.now() - t5;

        if (prob >= 0.70) {
          // Scale probability to weight: 0.70→0.60, 1.00→0.95 (capped at 0.95)
          const weight = Math.min(0.95, 0.60 + (prob - 0.70) * 1.167);
          allSignals.push({
            type: 'nn',
            label: `MobileNet classifier: ${(prob * 100).toFixed(1)}% AI`,
            weight,
          });
          maxScore = Math.max(maxScore, weight);
        }
        layer = 5;
      }
    } catch (err) {
      console.warn('[Lens SW] MobileNet inference failed:', err.message);
    }
  }

  // ── LAYER 6: Forensic NPR encoder ───────────────────────────────────────
  // Examines HOW the image was made (upsampling artifacts, frequency anomalies)
  // rather than WHAT is in it. Generalizes to unseen generators.
  if (imageData.width >= 64 && imageData.height >= 64) {
    const t6 = Date.now();
    try {
      const model = await getForensicModel();
      if (model) {
        const prob = tidy(() => {
          const pixels = applyNPR(imageData.data, imageData.width, imageData.height, 224);
          const input = tensor4d(pixels, [1, 224, 224, 3]);
          const output = model.predict(input);
          return output.dataSync()[0];
        });
        timing.l6 = Date.now() - t6;

        if (prob >= 0.50) {
          const weight = Math.min(0.95, prob - 0.05);
          allSignals.push({
            type: 'forensic',
            label: `Forensic encoder: ${(prob * 100).toFixed(1)}% AI`,
            weight,
          });
          maxScore = Math.max(maxScore, weight);
        }
        layer = 6;
      }
    } catch (err) {
      console.warn('[Lens SW] Forensic model inference failed:', err.message);
    }
  }

  return buildResult(url, maxScore, allSignals, layer, timing);
}

function buildResult(url, score, signals, layersCompleted, timing, error = null) {
  const interpretation = interpretScore(score, signals);
  return {
    url,
    score: Math.round(score * 100) / 100,
    confidence: Math.round(score * 100),
    interpretation,
    signals: signals.sort((a, b) => b.weight - a.weight),
    layersCompleted,
    timing,
    error,
    analyzedAt: Date.now(),
  };
}

function buildStats(urls) {
  const stats = { total: 0, definite: 0, likely: 0, possible: 0, clean: 0, pending: 0 };
  for (const url of urls) {
    stats.total++;
    const result = resultCache.get(url);
    if (!result) { stats.pending++; continue; }
    const level = result.interpretation?.level;
    if (level === 'definite') stats.definite++;
    else if (level === 'likely') stats.likely++;
    else if (level === 'possible') stats.possible++;
    else stats.clean++;
  }
  return stats;
}

// ---------------------------------------------------------------------------
// Debug logging
// ---------------------------------------------------------------------------

const LAYER_NAMES = { url: 'L1 URL', exif: 'L2 EXIF', xmp: 'L2 XMP', c2pa: 'L2 C2PA',
                      iptc: 'L2 IPTC', 'png-meta': 'L2 PNG', dom: 'DOM', pixel: 'L3 Pixel',
                      fft: 'L4 FFT', nn: 'L5 NN', forensic: 'L6 Forensic' };

function logResult(shortUrl, result) {
  // Push to debug ring buffer for popup debug tab
  const entry = {
    ts:      Date.now(),
    url:     shortUrl,
    verdict: result.interpretation?.level ?? '?',
    score:   result.score,
    layers:  result.layersCompleted,
    signals: (result.signals || []).map(s => ({
      type:   LAYER_NAMES[s.type] || s.type,
      label:  s.label,
      weight: s.weight,
    })),
    timing:  result.timing || {},
    error:   result.error || null,
  };
  debugLog.push(entry);
  if (debugLog.length > MAX_DEBUG_LOG) debugLog.shift();
  const { interpretation, score, signals, layersCompleted, timing, error } = result;
  const level = interpretation?.level ?? '?';
  const timingStr = Object.entries(timing || {}).map(([k, v]) => `${k}:${v}ms`).join(' ');

  if (signals && signals.length > 0) {
    const signalLines = signals
      .map(s => `      [${(LAYER_NAMES[s.type] || s.type).padEnd(8)}] ${s.label}  (weight: ${s.weight})`)
      .join('\n');
    console.log(
      `[Lens SW] ✦ ${shortUrl}\n` +
      `      verdict: ${level.toUpperCase()}  score: ${score}  layers: ${layersCompleted}  ${timingStr}\n` +
      `      signals (${signals.length}):\n${signalLines}` +
      (error ? `\n      ⚠ error: ${error}` : '')
    );
  } else {
    console.log(
      `[Lens SW] ✦ ${shortUrl} → ${level.toUpperCase()}  score: ${score}  layers: ${layersCompleted}  ${timingStr}  (no signals)` +
      (error ? `  ⚠ ${error}` : '')
    );
  }
}

// ---------------------------------------------------------------------------
// Image fetching — service worker can bypass CORS restrictions
// ---------------------------------------------------------------------------

async function fetchImageBytes(url) {
  if (url.startsWith('data:')) {
    // Data URL — decode inline
    const [header, b64] = url.split(',');
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  // Fetch with no-cors fallback
  let response;
  try {
    response = await fetch(url, { 
      mode: 'cors',
      credentials: 'omit',
      cache: 'force-cache',
    });
  } catch (_) {
    // CORS failed — try no-cors (opaque response, can still get bytes in SW)
    response = await fetch(url, { mode: 'no-cors', cache: 'force-cache' });
  }

  if (!response.ok && response.type !== 'opaque') {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.arrayBuffer();
}

// ---------------------------------------------------------------------------
// Image decoding via OffscreenCanvas
// ---------------------------------------------------------------------------

async function decodeImageToData(arrayBuffer) {
  const MAX_DIM = 512;

  // Use ImageDecoder API if available (Chrome 94+, Safari 16+)
  if (typeof ImageDecoder !== 'undefined') {
    try {
      const blob = new Blob([arrayBuffer]);
      const decoder = new ImageDecoder({ data: blob.stream(), type: sniffMimeType(arrayBuffer) });
      const { image } = await decoder.decode();
      const scale = Math.min(1, MAX_DIM / Math.max(image.displayWidth, image.displayHeight));
      const w = Math.floor(image.displayWidth * scale);
      const h = Math.floor(image.displayHeight * scale);
      const canvas = new OffscreenCanvas(w, h);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0, w, h);
      return ctx.getImageData(0, 0, w, h);
    } catch (_) {}
  }

  // Fallback: createImageBitmap
  const blob = new Blob([arrayBuffer]);
  const bitmap = await createImageBitmap(blob);

  const scale = Math.min(1, MAX_DIM / Math.max(bitmap.width, bitmap.height));
  const w = Math.floor(bitmap.width * scale);
  const h = Math.floor(bitmap.height * scale);

  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  return ctx.getImageData(0, 0, w, h);
}

function sniffMimeType(buffer) {
  const bytes = new Uint8Array(buffer, 0, 12);
  if (bytes[0] === 0xFF && bytes[1] === 0xD8) return 'image/jpeg';
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return 'image/png';
  if (bytes[0] === 0x52 && bytes[1] === 0x49) return 'image/webp';
  if (bytes[0] === 0x47 && bytes[1] === 0x49) return 'image/gif';
  return 'image/jpeg';
}

// ---------------------------------------------------------------------------
// Result cache management
// ---------------------------------------------------------------------------

function cacheResult(url, result) {
  if (resultCache.size >= MAX_CACHE_SIZE) {
    // Evict oldest entry
    const firstKey = resultCache.keys().next().value;
    resultCache.delete(firstKey);
  }
  resultCache.set(url, result);

  // Also persist to storage for popup access
  chrome.storage.session?.set({ [`lens_${hashUrl(url)}`]: result }).catch(() => {});
}

function hashUrl(url) {
  // Simple djb2 hash for storage keys
  let hash = 5381;
  for (let i = 0; i < Math.min(url.length, 200); i++) {
    hash = ((hash << 5) + hash) ^ url.charCodeAt(i);
    hash = hash >>> 0;
  }
  return hash.toString(36);
}

// ---------------------------------------------------------------------------
// Toolbar badge
// ---------------------------------------------------------------------------

function trackTabUrl(tabId, url) {
  if (!tabUrls.has(tabId)) tabUrls.set(tabId, new Set());
  tabUrls.get(tabId).add(url);
}

function updateTabBadge(tabId) {
  const urls = tabUrls.get(tabId);
  if (!urls) return;

  let flagged = 0;
  for (const url of urls) {
    const result = resultCache.get(url);
    if (!result) continue;
    const level = result.interpretation?.level;
    if (level === 'definite' || level === 'likely' || level === 'possible') flagged++;
  }

  const text   = flagged > 0 ? String(flagged) : '';
  const color  = flagged > 0 ? '#E53E3E' : '#4A5568';

  chrome.action.setBadgeText({ text, tabId }).catch(() => {});
  chrome.action.setBadgeBackgroundColor({ color, tabId }).catch(() => {});
}

// Clear badge and URL tracking when tab navigates or closes
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'loading') {
    tabUrls.delete(tabId);
    chrome.action.setBadgeText({ text: '', tabId }).catch(() => {});
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  tabUrls.delete(tabId);
});

console.log('[Lens SW] Service worker initialised');
