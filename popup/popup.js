/**
 * LENS — Popup Script
 */

const browser = globalThis.browser || globalThis.chrome;

// ── State ─────────────────────────────────────────────────────────────────

let currentFilter = 'flagged';
let pageImages    = [];
let results       = new Map(); // url → result
let settings      = {
  enabled:             true,
  minConfidenceToShow: 'possible',
  showOnHoverOnly:     false,
  highlightStyle:      'outline',
};
let flags = { disableL5: false };

// ── DOM refs ──────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const toggleEnabled = $('toggleEnabled');
const numScanning   = $('numScanning');
const numDefinite   = $('numDefinite');
const numLikely     = $('numLikely');
const numPossible   = $('numPossible');
const numClean      = $('numClean');
const statScanning  = $('statScanning');
const statDefinite  = $('statDefinite');
const statLikely    = $('statLikely');
const statPossible  = $('statPossible');
const statClean     = $('statClean');
const resultsList   = $('resultsList');
const emptyState    = $('emptyState');
const resultsPanel  = $('results');
const galleryPanel  = $('galleryPanel');
const galleryGrid   = $('galleryGrid');
const galleryEmpty  = $('galleryEmpty');
const settingsPanel = $('settingsPanel');
const footerUrl     = $('footerUrl');
const footerCache   = $('footerCacheCount');
const minConfidence = $('minConfidence');
const highlightStyle = $('highlightStyle');
const hoverOnly     = $('hoverOnly');
const clearCache       = $('clearCache');
const footerClearCache = $('footerClearCache');
const disableL5        = $('disableL5');
const debugPanel       = $('debugPanel');
const debugLogEl       = $('debugLog');
const debugCount       = $('debugCount');
const debugClear       = $('debugClear');
const debugCopy        = $('debugCopy');

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  await loadSettings();
  await loadFlags();
  applySettingsToUI();
  bindEvents();

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  try {
    const url = new URL(tab.url);
    footerUrl.textContent = url.hostname;
  } catch (_) {}

  try {
    const response = await browser.tabs.sendMessage(tab.id, { type: 'GET_PAGE_IMAGES' });
    if (response?.images) {
      // Dedup by URL in case the content script returned any duplicates
      const seen = new Set();
      pageImages = response.images.filter(i => !seen.has(i.src) && seen.add(i.src));
      numScanning.textContent = pageImages.length;
      statScanning.classList.toggle('is-nonzero', pageImages.length > 0);

      // Pre-populate results from data attrs the content script already wrote
      for (const img of pageImages) {
        if (img.level && img.confidence) {
          results.set(img.src, {
            confidence:     Number(img.confidence),
            score:          Number(img.confidence) / 100,
            interpretation: { level: img.level, label: img.label || img.level },
            signals:        [],
          });
        }
      }
      renderResults();
      renderGallery();
      updateStatsFromResults();
    }
  } catch (_) {
    emptyState.querySelector('.results__empty-text').textContent = 'No images found on this page.';
    emptyState.querySelector('.results__empty-icon').style.animation = 'none';
    return;
  }

  pollResults(tab.id, pageImages.map(i => i.src));
}

// ── Result polling ────────────────────────────────────────────────────────

let pollInterval;
let pollUrls = [];

async function pollResults(tabId, urls) {
  clearInterval(pollInterval);
  pollUrls = urls;

  const tick = async () => {
    // Re-fetch the URL list each tick — new images may have appeared via scroll/SPA
    try {
      const pageResp = await browser.tabs.sendMessage(tabId, { type: 'GET_PAGE_IMAGES' });
      if (pageResp?.images) {
        const seen = new Set();
        const fresh = pageResp.images.filter(i => !seen.has(i.src) && seen.add(i.src));
        // Merge any new images into pageImages without dropping existing entries
        for (const img of fresh) {
          if (!pageImages.some(i => i.src === img.src)) pageImages.push(img);
        }
        pollUrls = pageImages.map(i => i.src);
      }
    } catch (_) {}

    const response = await browser.runtime.sendMessage({ type: 'GET_TAB_STATS', urls: pollUrls }).catch(() => null);
    if (!response) return;

    // Merge any newly cached results into our local map
    if (response.results) {
      for (const [url, result] of Object.entries(response.results)) {
        results.set(url, result);
      }
    }

    updateStatsFromResults();
    renderResults();
    renderGallery();
    if (currentFilter === 'debug') renderDebugLog();
  };

  await tick();
  pollInterval = setInterval(tick, 800);
  // Keep polling for up to 2 minutes to handle infinite-scroll new images
  setTimeout(() => clearInterval(pollInterval), 120000);
}

// ── Stats ─────────────────────────────────────────────────────────────────

function updateStats(stats) {
  // Use SW-reported pending count directly; it knows what's actually in-flight.
  const pending = Math.max(0, stats.pending ?? (pageImages.length - (stats.definite + stats.likely + stats.possible + stats.clean)));

  numScanning.textContent = pending;
  numDefinite.textContent = stats.definite;
  numLikely.textContent   = stats.likely;
  numPossible.textContent = stats.possible;
  numClean.textContent    = stats.clean;

  statScanning.classList.toggle('is-nonzero', pending > 0);
  statDefinite.classList.toggle('is-nonzero', stats.definite > 0);
  statLikely.classList.toggle('is-nonzero',   stats.likely > 0);
  statPossible.classList.toggle('is-nonzero', stats.possible > 0);
  statClean.classList.toggle('is-nonzero',    stats.clean > 0);

  footerCache.textContent = `${stats.definite + stats.likely + stats.possible + stats.clean} cached`;
}

function updateStatsFromResults() {
  const stats = { definite: 0, likely: 0, possible: 0, clean: 0, pending: 0 };
  for (const { src } of pageImages) {
    const result = results.get(src);
    if (!result) { stats.pending++; continue; }
    const level = result.interpretation?.level;
    if      (level === 'definite') stats.definite++;
    else if (level === 'likely')   stats.likely++;
    else if (level === 'possible') stats.possible++;
    else                           stats.clean++;
  }
  updateStats(stats);
}

// ── Flagged list ──────────────────────────────────────────────────────────

function renderResults() {
  if (currentFilter !== 'flagged') return;

  const items = [];
  for (const img of pageImages) {
    const result = results.get(img.src);
    if (!result) continue;
    const level = result.interpretation?.level;
    if (level === 'clean' || level === 'unlikely') continue;
    items.push({ img, result, level });
  }
  items.sort((a, b) => b.result.score - a.result.score);

  if (items.length === 0) {
    const pending = pageImages.length - results.size;
    emptyState.querySelector('.results__empty-text').textContent = pending > 0
      ? `Analysing ${pending} image${pending > 1 ? 's' : ''}…`
      : 'No AI images detected on this page.';
    emptyState.querySelector('.results__empty-icon').textContent = pending > 0 ? '◈' : '✓';
    emptyState.hidden = false;
    resultsList.innerHTML = '';
    return;
  }

  emptyState.hidden = true;
  resultsList.innerHTML = '';
  for (const { img, result, level } of items.slice(0, 50)) {
    resultsList.appendChild(buildResultItem(img, result, level));
  }
}

function buildResultItem(img, result, level) {
  const li = document.createElement('li');
  li.className = `result-item result-item--${level}`;

  const filename    = img.src.split('/').pop()?.split('?')[0] || 'image';
  const displayUrl  = filename.length > 32 ? filename.slice(0, 30) + '…' : filename;
  const uniqueTypes = [...new Set((result.signals || []).slice(0, 3).map(s => s.type.toUpperCase()))];

  li.innerHTML = `
    <img class="result-item__thumb" src="${escapeAttr(img.src)}"
         alt="" loading="lazy" onerror="this.style.display='none'">
    <div class="result-item__info">
      <span class="result-item__url" title="${escapeAttr(img.src)}">${escapeHtml(displayUrl)}</span>
      <div class="result-item__signals">
        ${uniqueTypes.map(t => `<span class="result-item__signal-tag">${escapeHtml(t)}</span>`).join('')}
      </div>
    </div>
    <div class="result-item__badge">
      <span class="result-item__level">${escapeHtml(result.interpretation?.label || level)}</span>
      <span class="result-item__confidence">${result.confidence}%</span>
    </div>
  `;

  li.addEventListener('click', () => browser.tabs.create({ url: img.src, active: false }));
  li.style.cursor = 'pointer';
  li.title = `Open image · ${result.interpretation?.label} (${result.confidence}%)`;
  return li;
}

// ── Image gallery ─────────────────────────────────────────────────────────

function renderGallery() {
  if (currentFilter !== 'images') return;

  const items = pageImages.map(img => ({
    img,
    result: results.get(img.src) || null,
  }));

  if (items.length === 0) {
    galleryEmpty.hidden = false;
    galleryGrid.innerHTML = '';
    return;
  }

  galleryEmpty.hidden = true;
  galleryGrid.innerHTML = '';

  for (const { img, result } of items) {
    galleryGrid.appendChild(buildGalleryCard(img, result));
  }
}

function buildGalleryCard(img, result) {
  const card = document.createElement('div');
  const level = result?.interpretation?.level || 'pending';
  card.className = `gallery-card gallery-card--${level}`;

  const confidence = result ? `${result.confidence}%` : '…';
  const label      = result ? (result.interpretation?.label || level) : 'Scanning';

  card.innerHTML = `
    <div class="gallery-card__img-wrap">
      <img class="gallery-card__img" src="${escapeAttr(img.src)}"
           alt="${escapeAttr(img.alt || '')}" loading="lazy"
           onerror="this.parentElement.classList.add('gallery-card__img-wrap--error')">
      <div class="gallery-card__overlay">
        <span class="gallery-card__label">${escapeHtml(label)}</span>
        <span class="gallery-card__confidence">${escapeHtml(confidence)}</span>
      </div>
    </div>
  `;

  card.addEventListener('click', () => browser.tabs.create({ url: img.src, active: false }));
  card.title = result
    ? `${result.interpretation?.label} · ${result.confidence}% — click to open`
    : 'Scanning…';

  return card;
}

// ── Events ────────────────────────────────────────────────────────────────

function bindEvents() {
  toggleEnabled.addEventListener('change', async () => {
    settings.enabled = toggleEnabled.checked;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  document.querySelectorAll('.filter').forEach(btn => {
    btn.addEventListener('click', () => {
      currentFilter = btn.dataset.filter;
      document.querySelectorAll('.filter').forEach(b => {
        b.classList.toggle('filter--active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      });

      resultsPanel.hidden  = currentFilter !== 'flagged';
      galleryPanel.hidden  = currentFilter !== 'images';
      settingsPanel.hidden = currentFilter !== 'settings';
      debugPanel.hidden    = currentFilter !== 'debug';

      if (currentFilter === 'flagged') renderResults();
      if (currentFilter === 'images')  renderGallery();
      if (currentFilter === 'debug')   renderDebugLog();
    });
  });

  minConfidence.addEventListener('change', async () => {
    settings.minConfidenceToShow = minConfidence.value;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  highlightStyle.addEventListener('change', async () => {
    settings.highlightStyle = highlightStyle.value;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  hoverOnly.addEventListener('change', async () => {
    settings.showOnHoverOnly = hoverOnly.checked;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  async function doClearCache() {
    await browser.runtime.sendMessage({ type: 'CLEAR_CACHE' });
    results.clear();
    updateStatsFromResults();
    renderResults();
    renderGallery();
    renderDebugLog();
    clearCache.textContent = 'Cache cleared ✓';
    footerClearCache.textContent = '✓';
    setTimeout(() => {
      clearCache.textContent = 'Clear analysis cache';
      footerClearCache.textContent = '↺';
    }, 2000);
  }
  clearCache.addEventListener('click', doClearCache);
  footerClearCache.addEventListener('click', doClearCache);

  disableL5.addEventListener('change', async () => {
    flags.disableL5 = disableL5.checked;
    await browser.runtime.sendMessage({ type: 'SET_FLAGS', flags });
  });

  debugClear.addEventListener('click', async () => {
    await browser.runtime.sendMessage({ type: 'CLEAR_CACHE' });
    results.clear();
    renderDebugLog();
  });

  debugCopy.addEventListener('click', async () => {
    const resp = await browser.runtime.sendMessage({ type: 'GET_DEBUG_LOG' }).catch(() => null);
    if (!resp?.log) return;
    const text = resp.log.map(formatDebugEntry).join('\n\n');
    navigator.clipboard.writeText(text).catch(() => {});
    debugCopy.textContent = 'Copied ✓';
    setTimeout(() => { debugCopy.textContent = 'Copy'; }, 2000);
  });
}

// ── Settings ──────────────────────────────────────────────────────────────

async function loadSettings() {
  return new Promise(resolve => {
    browser.storage.sync.get('lensSettings', (data) => {
      if (data.lensSettings) settings = { ...settings, ...data.lensSettings };
      resolve();
    });
  });
}

async function loadFlags() {
  return new Promise(resolve => {
    browser.storage.sync.get('lensFlags', (data) => {
      if (data.lensFlags) flags = { ...flags, ...data.lensFlags };
      resolve();
    });
  });
}

async function saveSettings() {
  return browser.storage.sync.set({ lensSettings: settings });
}

function applySettingsToUI() {
  toggleEnabled.checked   = settings.enabled;
  minConfidence.value     = settings.minConfidenceToShow;
  highlightStyle.value    = settings.highlightStyle;
  hoverOnly.checked       = settings.showOnHoverOnly;
  disableL5.checked       = flags.disableL5;
}

async function broadcastSettingsUpdate() {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  browser.tabs.sendMessage(tab.id, { type: 'SETTINGS_UPDATED', settings }).catch(() => {});
}

// ── Helpers ───────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escapeAttr(str) {
  return String(str).replace(/"/g,'%22').replace(/'/g,'%27');
}

// ── Debug log ─────────────────────────────────────────────────────────────

function formatDebugEntry(e) {
  const time = new Date(e.ts).toLocaleTimeString();
  const signals = e.signals.length
    ? e.signals.map(s => `  ${s.type.padEnd(14)} ${s.label}  (${(s.weight * 100).toFixed(1)}%)`).join('\n')
    : '  (no signals)';
  const timing = Object.entries(e.timing).map(([k, v]) => `${k}:${v}ms`).join(' ');
  return `[${time}] ${e.verdict.toUpperCase()}  score:${e.score?.toFixed(3)}  layers:${e.layers}  ${timing}\n${e.url}\n${signals}${e.error ? `\n  ⚠ ${e.error}` : ''}`;
}

async function renderDebugLog() {
  const resp = await browser.runtime.sendMessage({ type: 'GET_DEBUG_LOG' }).catch(() => null);
  const log = resp?.log || [];

  debugCount.textContent = `${log.length} result${log.length !== 1 ? 's' : ''}`;

  if (log.length === 0) {
    debugLogEl.innerHTML = '<div class="debug-panel__empty">No results yet. Browse pages to populate.</div>';
    return;
  }

  const VERDICT_CLASS = { definite: 'definite', likely: 'likely', possible: 'possible', unlikely: 'clean', clean: 'clean' };

  debugLogEl.innerHTML = '';
  for (const e of [...log].reverse()) {
    const div = document.createElement('div');
    div.className = `debug-entry debug-entry--${VERDICT_CLASS[e.verdict] || 'clean'}`;

    const time    = new Date(e.ts).toLocaleTimeString();
    const timing  = Object.entries(e.timing).map(([k, v]) => `${k}:${v}ms`).join(' ');
    const signals = e.signals.map(s =>
      `<div class="debug-entry__signal"><span class="debug-entry__sig-type">${escapeHtml(s.type)}</span> ${escapeHtml(s.label)} <span class="debug-entry__sig-weight">${(s.weight * 100).toFixed(1)}%</span></div>`
    ).join('');

    div.innerHTML = `
      <div class="debug-entry__header">
        <span class="debug-entry__verdict debug-entry__verdict--${VERDICT_CLASS[e.verdict] || 'clean'}">${e.verdict.toUpperCase()}</span>
        <span class="debug-entry__score">${e.score?.toFixed(3)}</span>
        <span class="debug-entry__time">${time}</span>
      </div>
      <div class="debug-entry__url" title="${escapeAttr(e.url)}">${escapeHtml(e.url)}</div>
      <div class="debug-entry__timing">${escapeHtml(timing)}</div>
      ${signals}
      ${e.error ? `<div class="debug-entry__error">⚠ ${escapeHtml(e.error)}</div>` : ''}
    `;
    debugLogEl.appendChild(div);
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────

init().catch(console.error);
