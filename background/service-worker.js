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

// In-memory result cache: URL hash → result
// Survives across content script navigations within a browser session
const resultCache = new Map();
const MAX_CACHE_SIZE = 500;

// Concurrent analysis throttle — don't hammer all images at once
const analysisQueue = new Map(); // url → Promise
const MAX_CONCURRENT = 4;
let activeCount = 0;

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

    // Check cache for immediate inline response
    if (resultCache.has(url)) {
      console.log(`[Lens SW] Cache hit: ${shortUrl}`);
      sendResponse({ result: resultCache.get(url), fromCache: true });
      return false;
    }

    // Use event.waitUntil pattern via a port keepalive to prevent SW termination
    // during async fetch+decode (MV3 SW can be suspended mid-flight otherwise)
    const keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(() => {}), 20000);

    analyzeImage(url, domSignals || {})
      .then(result => {
        cacheResult(url, result);
        logResult(shortUrl, result);
        sendResponse({ result, fromCache: false });
      })
      .catch(err => {
        console.warn(`[Lens SW] Analysis error for ${shortUrl}:`, err.message);
        sendResponse({ error: err.message });
      })
      .finally(() => clearInterval(keepAlive));

    return true; // Keep message channel open for async response
  }

  if (message.type === 'GET_TAB_STATS') {
    // Collect stats for all cached results from the active tab
    const stats = buildStats(message.urls || []);
    sendResponse({ stats });
    return true;
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
  const pixelResult = analyzePixelStatistics(imageData);
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
                      iptc: 'L2 IPTC', 'png-meta': 'L2 PNG', dom: 'DOM', pixel: 'L3 Pixel', fft: 'L4 FFT' };

function logResult(shortUrl, result) {
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
  // Use ImageDecoder API if available (Chrome 94+, Safari 16+)
  if (typeof ImageDecoder !== 'undefined') {
    try {
      const blob = new Blob([arrayBuffer]);
      const decoder = new ImageDecoder({ data: blob.stream(), type: sniffMimeType(arrayBuffer) });
      const { image } = await decoder.decode();
      const canvas = new OffscreenCanvas(image.displayWidth, image.displayHeight);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      return ctx.getImageData(0, 0, canvas.width, canvas.height);
    } catch (_) {}
  }

  // Fallback: createImageBitmap
  const blob = new Blob([arrayBuffer]);
  const bitmap = await createImageBitmap(blob);

  // Cap size to avoid OOM
  const maxDim = 1024;
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
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

console.log('[Lens SW] Service worker initialised');
