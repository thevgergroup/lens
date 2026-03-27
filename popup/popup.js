/**
 * LENS — Popup Script
 * Queries the active tab for images, displays analysis results, manages settings.
 */

// Chrome/Firefox API compat shim
const browser = globalThis.browser || globalThis.chrome;

// ── State ────────────────────────────────────────────────────────────────

let currentFilter = 'flagged';
let pageImages = [];
let results = new Map(); // url → result
let settings = {
  enabled: true,
  minConfidenceToShow: 'possible',
  showOnHoverOnly: false,
  badgePosition: 'top-left',
};

// ── DOM refs ─────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const toggleEnabled   = $('toggleEnabled');
const numScanning     = $('numScanning');
const numDefinite     = $('numDefinite');
const numLikely       = $('numLikely');
const numPossible     = $('numPossible');
const numClean        = $('numClean');
const statScanning    = $('statScanning');
const statDefinite    = $('statDefinite');
const statLikely      = $('statLikely');
const statPossible    = $('statPossible');
const statClean       = $('statClean');
const resultsList     = $('resultsList');
const emptyState      = $('emptyState');
const resultsPanel    = $('results');
const settingsPanel   = $('settingsPanel');
const footerUrl       = $('footerUrl');
const footerCache     = $('footerCacheCount');
const minConfidence   = $('minConfidence');
const hoverOnly       = $('hoverOnly');
const badgePosition   = $('badgePosition');
const clearCache      = $('clearCache');

// ── Init ─────────────────────────────────────────────────────────────────

async function init() {
  await loadSettings();
  applySettingsToUI();
  bindEvents();

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  // Show hostname in footer
  try {
    const url = new URL(tab.url);
    footerUrl.textContent = url.hostname;
  } catch (_) {}

  // Ask content script for page images
  try {
    const response = await browser.tabs.sendMessage(tab.id, { type: 'GET_PAGE_IMAGES' });
    if (response?.images) {
      pageImages = response.images;
      numScanning.textContent = pageImages.length;
      statScanning.classList.toggle('is-nonzero', pageImages.length > 0);
    }
  } catch (_) {
    // Content script may not be injected yet (e.g. browser internal page)
    emptyState.querySelector('.results__empty-text').textContent = 
      'No images found on this page.';
    emptyState.querySelector('.results__empty-icon').style.animation = 'none';
    return;
  }

  // Ask service worker for cached results for these URLs
  const imageUrls = pageImages.map(i => i.src);
  const swResponse = await browser.runtime.sendMessage({
    type: 'GET_TAB_STATS',
    urls: imageUrls,
  });

  if (swResponse?.stats) {
    updateStats(swResponse.stats);
  }

  // Poll for new results
  pollResults(tab.id, imageUrls);
}

// ── Result polling ────────────────────────────────────────────────────────

let pollInterval;

async function pollResults(tabId, urls) {
  clearInterval(pollInterval);

  const fetchResults = async () => {
    const response = await browser.runtime.sendMessage({
      type: 'GET_TAB_STATS',
      urls,
    }).catch(() => null);

    if (!response?.stats) return;

    updateStats(response.stats);

    // Check if all images are analyzed
    if (response.stats.pending === 0) {
      clearInterval(pollInterval);
    }
  };

  await fetchResults();
  pollInterval = setInterval(fetchResults, 600);

  // Stop after 30 seconds regardless
  setTimeout(() => clearInterval(pollInterval), 30000);
}

// Listen for live results pushed from content script via service worker
browser.runtime.onMessage.addListener((message) => {
  if (message.type === 'ANALYSIS_RESULT') {
    const { url, result } = message;
    results.set(url, result);
    renderResults();
    updateStatsFromResults();
  }
});

// ── Render ─────────────────────────────────────────────────────────────────

function updateStats(stats) {
  const total = pageImages.length;
  const pending = total - (stats.definite + stats.likely + stats.possible + stats.clean);

  numScanning.textContent = Math.max(0, pending);
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
  for (const url of pageImages.map(i => i.src)) {
    const result = results.get(url);
    if (!result) { stats.pending++; continue; }
    const level = result.interpretation?.level;
    if (level === 'definite') stats.definite++;
    else if (level === 'likely') stats.likely++;
    else if (level === 'possible') stats.possible++;
    else stats.clean++;
  }
  updateStats(stats);
}

function renderResults() {
  const items = [];

  for (const img of pageImages) {
    const result = results.get(img.src);
    if (!result) continue;

    const level = result.interpretation?.level;
    if (currentFilter === 'flagged') {
      if (level === 'clean' || level === 'unlikely') continue;
    }

    items.push({ img, result, level });
  }

  // Sort: highest confidence first
  items.sort((a, b) => b.result.score - a.result.score);

  if (items.length === 0) {
    const pending = pageImages.length - results.size;
    if (pending > 0) {
      emptyState.querySelector('.results__empty-text').textContent = `Analysing ${pending} image${pending > 1 ? 's' : ''}…`;
    } else {
      emptyState.querySelector('.results__empty-text').textContent = 
        currentFilter === 'flagged' 
          ? 'No AI images detected on this page.' 
          : 'No images analysed yet.';
      emptyState.querySelector('.results__empty-icon').textContent = '✓';
      emptyState.querySelector('.results__empty-icon').style.animation = 'none';
    }
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

  const filename = img.src.split('/').pop()?.split('?')[0] || 'image';
  const displayUrl = filename.length > 30 ? filename.slice(0, 28) + '…' : filename;

  const topSignals = (result.signals || [])
    .slice(0, 3)
    .map(s => s.type.toUpperCase());

  const uniqueTypes = [...new Set(topSignals)];

  li.innerHTML = `
    <img class="result-item__thumb" src="${escapeAttr(img.src)}" 
         alt="" loading="lazy" 
         onerror="this.style.display='none'">
    <div class="result-item__info">
      <span class="result-item__url" title="${escapeAttr(img.src)}">${escapeHtml(displayUrl)}</span>
      <div class="result-item__signals">
        ${uniqueTypes.map(t => `<span class="result-item__signal-tag">${escapeHtml(t)}</span>`).join('')}
      </div>
    </div>
    <div class="result-item__badge">
      <span class="result-item__level">${escapeHtml(result.interpretation?.level || '')}</span>
      <span class="result-item__confidence">${result.confidence}%</span>
    </div>
  `;

  // Click to open image in new tab
  li.addEventListener('click', () => {
    browser.tabs.create({ url: img.src, active: false });
  });
  li.style.cursor = 'pointer';
  li.title = `Open image · ${result.interpretation?.label} (${result.confidence}%)`;

  return li;
}

// ── Events ────────────────────────────────────────────────────────────────

function bindEvents() {
  // Enable/disable toggle
  toggleEnabled.addEventListener('change', async () => {
    settings.enabled = toggleEnabled.checked;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  // Filter tabs
  document.querySelectorAll('.filter').forEach(btn => {
    btn.addEventListener('click', () => {
      currentFilter = btn.dataset.filter;
      document.querySelectorAll('.filter').forEach(b => {
        b.classList.toggle('filter--active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      });

      if (currentFilter === 'settings') {
        resultsPanel.hidden = true;
        settingsPanel.hidden = false;
      } else {
        resultsPanel.hidden = false;
        settingsPanel.hidden = true;
        renderResults();
      }
    });
  });

  // Settings controls
  minConfidence.addEventListener('change', async () => {
    settings.minConfidenceToShow = minConfidence.value;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  hoverOnly.addEventListener('change', async () => {
    settings.showOnHoverOnly = hoverOnly.checked;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  badgePosition.addEventListener('change', async () => {
    settings.badgePosition = badgePosition.value;
    await saveSettings();
    broadcastSettingsUpdate();
  });

  clearCache.addEventListener('click', async () => {
    await browser.runtime.sendMessage({ type: 'CLEAR_CACHE' });
    results.clear();
    renderResults();
    clearCache.textContent = 'Cache cleared ✓';
    setTimeout(() => { clearCache.textContent = 'Clear analysis cache'; }, 2000);
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

async function saveSettings() {
  return browser.storage.sync.set({ lensSettings: settings });
}

function applySettingsToUI() {
  toggleEnabled.checked    = settings.enabled;
  minConfidence.value      = settings.minConfidenceToShow;
  hoverOnly.checked        = settings.showOnHoverOnly;
  badgePosition.value      = settings.badgePosition;
}

async function broadcastSettingsUpdate() {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  browser.tabs.sendMessage(tab.id, {
    type: 'SETTINGS_UPDATED',
    settings,
  }).catch(() => {});
}

// ── Helpers ───────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function escapeAttr(str) {
  return String(str).replace(/"/g, '%22').replace(/'/g, '%27');
}

// ── Boot ──────────────────────────────────────────────────────────────────

init().catch(console.error);
