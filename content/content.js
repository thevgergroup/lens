/**
 * LENS — Content Script
 * Observes page images, sends them for analysis, marks results via data attributes.
 * No DOM wrapping — uses CSS outline/filter on the <img> directly.
 */

(function () {
  'use strict';

  if (window.__lensInjected) return;
  window.__lensInjected = true;

  const MIN_SIZE_PX  = 150;  // ignore small icons/logos
  const MIN_NATURAL  = 100;  // ignore tiny natural-size images
  const MAX_ASPECT   = 5;    // ignore banners (very wide/thin strips)

  // Structural containers whose images are almost certainly UI chrome
  const SKIP_ANCESTORS = ['header', 'nav', 'footer', 'aside'];

  // Alt/aria text patterns that signal logos/icons/avatars
  const SKIP_ALT_RE = /\b(logo|icon|avatar|badge|button|sprite|emoji|flag|seal|crest)\b/i;

  const processedImages = new WeakSet();

  // ---------------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------------

  let settings = {
    enabled: true,
    minConfidenceToShow: 'possible',
    showOnHoverOnly: false,
    highlightStyle: 'outline',  // 'outline' | 'desaturate' | 'both'
  };

  chrome.storage.sync.get('lensSettings', (data) => {
    if (data.lensSettings) settings = { ...settings, ...data.lensSettings };
  });

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'SETTINGS_UPDATED') applySettings(message.settings);
    if (message.type === 'ANALYSIS_RESULT')  handleResult(message.url, message.result);
    if (message.type === 'REANALYZE') {
      // Clear processed set and remove all badges so images get re-analyzed from scratch
      document.querySelectorAll('img[data-lens-level]').forEach(img => {
        img.removeAttribute('data-lens-level');
        img.removeAttribute('data-lens-confidence');
        processedImages.delete(img);
        const wrapper = img.closest('.lens-wrapper');
        if (wrapper) wrapper.querySelector('.lens-badge')?.remove();
      });
      // Re-queue all visible images
      document.querySelectorAll('img').forEach(img => {
        if (shouldAnalyze(img)) queueImage(img);
      });
    }
  });

  function applySettings(newSettings) {
    settings = { ...settings, ...newSettings };
    document.querySelectorAll('img[data-lens-level]').forEach(img => {
      updateHighlight(img, img.getAttribute('data-lens-level'));
    });
  }

  // ---------------------------------------------------------------------------
  // Filtering — decide whether an image is worth analysing
  // ---------------------------------------------------------------------------

  function shouldAnalyze(img, { allowProcessed = false } = {}) {
    if (!img?.src) return false;
    if (img.src.startsWith('chrome-extension://') || img.src.startsWith('moz-extension://')) return false;
    if (img.src.startsWith('data:image/gif;base64,R0lGOD')) return false; // 1px GIF tracker
    if (!allowProcessed && processedImages.has(img)) return false;

    // Rendered size
    const rect = img.getBoundingClientRect();
    if (rect.width < MIN_SIZE_PX || rect.height < MIN_SIZE_PX) return false;

    // Natural size (if loaded)
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (nw > 0 && nw < MIN_NATURAL) return false;
    if (nh > 0 && nh < MIN_NATURAL) return false;

    // Extreme aspect ratio — banners, dividers, UI strips
    if (nw > 0 && nh > 0) {
      const ratio = Math.max(nw / nh, nh / nw);
      if (ratio > MAX_ASPECT) return false;
    }

    // Structural UI chrome
    if (SKIP_ANCESTORS.some(tag => img.closest(tag))) return false;

    // Alt text / aria-label signals logo/icon
    const altText = (img.alt || img.getAttribute('aria-label') || '').trim();
    if (altText && SKIP_ALT_RE.test(altText)) return false;

    // role="presentation" or role="none" → decorative
    const role = img.getAttribute('role');
    if (role === 'presentation' || role === 'none') return false;

    // CSS background-image lookalikes with display:none
    if (getComputedStyle(img).display === 'none') return false;

    return true;
  }

  // ---------------------------------------------------------------------------
  // Queue & analyse
  // ---------------------------------------------------------------------------

  function queueImage(img) {
    if (!shouldAnalyze(img)) return;
    processedImages.add(img);

    const domSignals = {
      altText:        img.alt || '',
      title:          img.title || '',
      ariaLabel:      img.getAttribute('aria-label') || '',
      figcaptionText: img.closest('figure')?.querySelector('figcaption')?.textContent?.trim() || '',
    };

    // Use Promise form — Firefox drops the callback channel on async SW responses.
    // Results come back either via the resolved promise or via ANALYSIS_RESULT push.
    chrome.runtime.sendMessage({ type: 'ANALYZE_IMAGE', payload: { url: img.src, domSignals } })
      .then(response => { if (response?.result) handleResult(img.src, response.result); })
      .catch(() => {}); // SW will push ANALYSIS_RESULT if the channel closes early
  }

  // ---------------------------------------------------------------------------
  // Result handling — mark image with data attribute, apply CSS highlight
  // ---------------------------------------------------------------------------

  function handleResult(url, result) {
    document.querySelectorAll('img').forEach(img => {
      if (img.src !== url) return;

      const level = result.interpretation?.level;
      img.setAttribute('data-lens-level',      level);
      img.setAttribute('data-lens-confidence', result.confidence);
      img.setAttribute('data-lens-label',      result.interpretation?.label || '');

      if (shouldHighlight(result)) {
        updateHighlight(img, level);
      } else {
        clearHighlight(img);
      }
    });
  }

  function shouldHighlight(result) {
    if (!settings.enabled) return false;
    const level = result.interpretation?.level;
    switch (settings.minConfidenceToShow) {
      case 'definite': return level === 'definite';
      case 'likely':   return level === 'definite' || level === 'likely';
      case 'possible': return level !== 'clean' && level !== 'unlikely';
      case 'all':      return true;
      default:         return level !== 'clean' && level !== 'unlikely';
    }
  }

  function updateHighlight(img, level) {
    if (!settings.enabled) { clearHighlight(img); return; }
    img.setAttribute('data-lens-highlight', settings.highlightStyle || 'outline');
    img.setAttribute('data-lens-hover-only', settings.showOnHoverOnly ? 'true' : 'false');
  }

  function clearHighlight(img) {
    img.removeAttribute('data-lens-highlight');
  }

  // ---------------------------------------------------------------------------
  // Observation
  // ---------------------------------------------------------------------------

  const intersectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const img = entry.target;
      intersectionObserver.unobserve(img);
      if (img.complete && img.naturalWidth > 0) {
        queueImage(img);
      } else {
        img.addEventListener('load', () => queueImage(img), { once: true });
      }
    });
  }, { rootMargin: '200px', threshold: 0 });

  function observeImage(img) {
    if (processedImages.has(img)) return;
    if (!img.src) return;
    intersectionObserver.observe(img);
  }

  function scanPage() {
    if (!settings.enabled) return;
    document.querySelectorAll('img[src]').forEach(observeImage);
  }

  const mutationObserver = new MutationObserver((mutations) => {
    if (!settings.enabled) return;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.tagName === 'IMG' && node.src) observeImage(node);
        node.querySelectorAll?.('img[src]').forEach(observeImage);
      }
      if (mutation.type === 'attributes' && mutation.target.tagName === 'IMG') {
        const img = mutation.target;
        processedImages.delete(img);
        clearHighlight(img);
        observeImage(img);
      }
    }
  });

  mutationObserver.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src'],
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scanPage);
  } else {
    scanPage();
  }

  window.addEventListener('load', () => setTimeout(scanPage, 500));

  // ---------------------------------------------------------------------------
  // Popup communication
  // ---------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'GET_PAGE_IMAGES') {
      // Report images that either passed our filter or already have a result.
      // Dedup by URL — CDN variants with different query params count as one entry.
      const seen = new Set();
      const images = Array.from(document.querySelectorAll('img[src]'))
        .filter(img => {
          if (!img.src || seen.has(img.src)) return false;
          // Include if already analysed or eligible for analysis
          if (img.hasAttribute('data-lens-level')) { seen.add(img.src); return true; }
          if (shouldAnalyze(img, { allowProcessed: true })) { seen.add(img.src); return true; }
          return false;
        })
        .map(img => ({
          src:        img.src,
          width:      img.naturalWidth,
          height:     img.naturalHeight,
          alt:        img.alt,
          level:      img.getAttribute('data-lens-level') || null,
          confidence: img.getAttribute('data-lens-confidence') || null,
          label:      img.getAttribute('data-lens-label') || null,
        }))
        .slice(0, 200);
      sendResponse({ images });
      return false;
    }

    if (message.type === 'TOGGLE_ENABLED') {
      settings.enabled = message.enabled;
      if (!settings.enabled) {
        document.querySelectorAll('img[data-lens-highlight]').forEach(clearHighlight);
      } else {
        document.querySelectorAll('img[data-lens-level]').forEach(img => {
          updateHighlight(img, img.getAttribute('data-lens-level'));
        });
        scanPage();
      }
      return false;
    }
  });

})();
