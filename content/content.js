/**
 * LENS — Content Script
 * Observes page images, sends them for analysis, injects result badges.
 * Runs in page context — communicates with Service Worker via chrome.runtime.
 */

(function () {
  'use strict';

  // Avoid double-injection
  if (window.__lensInjected) return;
  window.__lensInjected = true;

  const BADGE_CLASS = 'lens-badge';
  const WRAPPER_CLASS = 'lens-wrapper';
  const MIN_IMAGE_SIZE = 80; // px — ignore tiny icons/favicons

  // Track which images we've already processed
  const processedImages = new WeakSet();
  const pendingImages = new Map(); // img element → request id
  let requestIdCounter = 0;

  // ---------------------------------------------------------------------------
  // Message channel to service worker
  // ---------------------------------------------------------------------------

  // Chrome: use chrome.runtime.sendMessage
  // Firefox: same API
  function sendToSW(type, payload) {
    return new Promise((resolve, reject) => {
      const id = ++requestIdCounter;
      chrome.runtime.sendMessage({ type, payload: { ...payload }, id }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(response);
        }
      });
    });
  }

  // Listen for analysis results pushed back from service worker
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'ANALYSIS_RESULT') {
      handleResult(message.url, message.result);
    }
    if (message.type === 'SETTINGS_UPDATED') {
      applySettings(message.settings);
    }
  });

  // ---------------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------------

  let settings = {
    enabled: true,
    showOnAllImages: true,
    minConfidenceToShow: 'possible', // 'definite' | 'likely' | 'possible' | 'all'
    badgePosition: 'top-left',
    showOnHoverOnly: false,
  };

  chrome.storage.sync.get('lensSettings', (data) => {
    if (data.lensSettings) settings = { ...settings, ...data.lensSettings };
  });

  function applySettings(newSettings) {
    settings = { ...settings, ...newSettings };
    // Re-evaluate badge visibility
    document.querySelectorAll(`.${BADGE_CLASS}`).forEach(badge => {
      updateBadgeVisibility(badge);
    });
  }

  // ---------------------------------------------------------------------------
  // Image discovery & queuing
  // ---------------------------------------------------------------------------

  function shouldAnalyze(img) {
    if (!img || !img.src) return false;
    if (img.src.startsWith('chrome-extension://')) return false;
    if (img.src.startsWith('moz-extension://')) return false;
    if (processedImages.has(img)) return false;

    // Size check — wait for image to load to get dimensions
    const rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth || 0;
    const naturalH = img.naturalHeight || 0;

    // Must be a real image, not a 1px tracker or icon
    if (rect.width < MIN_IMAGE_SIZE && rect.height < MIN_IMAGE_SIZE) return false;
    if (naturalW > 0 && naturalW < 32) return false;
    if (naturalH > 0 && naturalH < 32) return false;

    return true;
  }

  function queueImage(img) {
    if (!shouldAnalyze(img)) return;
    processedImages.add(img);

    // Wrap image in position:relative container for badge overlay
    wrapImage(img);

    // Gather DOM signals to help with analysis
    const domSignals = {
      altText: img.alt || '',
      title: img.title || '',
      ariaLabel: img.getAttribute('aria-label') || '',
      figcaptionText: img.closest('figure')?.querySelector('figcaption')?.textContent?.trim() || '',
    };

    // Send to service worker for analysis
    console.log('[Lens] Sending ANALYZE_IMAGE for', img.src.split('/').pop());
    sendToSW('ANALYZE_IMAGE', { url: img.src, domSignals })
      .then(response => {
        console.log('[Lens] Response for', img.src.split('/').pop(), JSON.stringify(response)?.slice(0, 100));
        if (response?.result) {
          handleResult(img.src, response.result);
        }
      })
      .catch(err => {
        console.error('[Lens] Analysis failed for', img.src.slice(0, 60), err.message);
      });
  }

  // ---------------------------------------------------------------------------
  // DOM wrapping & badge injection
  // ---------------------------------------------------------------------------

  function wrapImage(img) {
    // Don't double-wrap
    if (img.parentElement?.classList.contains(WRAPPER_CLASS)) return;
    if (img.closest(`.${WRAPPER_CLASS}`)) return;

    const wrapper = document.createElement('div');
    wrapper.className = WRAPPER_CLASS;
    wrapper.style.cssText = `
      position: relative !important;
      display: inline-block !important;
      line-height: 0 !important;
    `;

    img.parentNode.insertBefore(wrapper, img);
    wrapper.appendChild(img);

    // Add a pending indicator
    const pending = document.createElement('div');
    pending.className = `${BADGE_CLASS} ${BADGE_CLASS}--pending`;
    pending.setAttribute('aria-label', 'Analysing image…');
    wrapper.appendChild(pending);
  }

  function handleResult(url, result) {
    // Find all images matching this URL — use a loop since CSS.escape mangles full URLs
    const images = Array.from(document.querySelectorAll('img')).filter(img => img.src === url);

    images.forEach(img => {
      const wrapper = img.closest(`.${WRAPPER_CLASS}`);
      if (!wrapper) return;

      // Remove pending badge
      wrapper.querySelectorAll(`.${BADGE_CLASS}--pending`).forEach(el => el.remove());

      // Only show badge if above confidence threshold
      if (!shouldShowBadge(result)) return;

      injectBadge(wrapper, result);
    });
  }

  function shouldShowBadge(result) {
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

  function injectBadge(wrapper, result) {
    // Remove existing badge
    wrapper.querySelectorAll(`.${BADGE_CLASS}:not(.${BADGE_CLASS}--pending)`).forEach(el => el.remove());

    const { interpretation, confidence, signals } = result;

    const badge = document.createElement('div');
    badge.className = `${BADGE_CLASS} ${BADGE_CLASS}--${interpretation.level}`;
    badge.setAttribute('data-lens-level', interpretation.level);
    badge.setAttribute('data-lens-confidence', confidence);
    badge.setAttribute('aria-label', `${interpretation.label} (${confidence}% confidence)`);
    badge.setAttribute('role', 'status');

    // Badge inner HTML
    const topSignal = signals?.[0];
    badge.innerHTML = `
      <span class="${BADGE_CLASS}__icon" aria-hidden="true">${getLevelIcon(interpretation.level)}</span>
      <span class="${BADGE_CLASS}__label">${interpretation.label}</span>
      <span class="${BADGE_CLASS}__confidence">${confidence}%</span>
      ${topSignal ? `<div class="${BADGE_CLASS}__tooltip">
        <div class="${BADGE_CLASS}__tooltip-title">${interpretation.label} · ${confidence}% confidence</div>
        <div class="${BADGE_CLASS}__tooltip-signals">
          ${signals.slice(0, 4).map(s => `
            <div class="${BADGE_CLASS}__signal">
              <span class="${BADGE_CLASS}__signal-type">${s.type.toUpperCase()}</span>
              <span class="${BADGE_CLASS}__signal-label">${escapeHtml(s.label)}</span>
            </div>
          `).join('')}
          ${signals.length > 4 ? `<div class="${BADGE_CLASS}__signal-more">+${signals.length - 4} more signals</div>` : ''}
        </div>
      </div>` : ''}
    `;

    // Position badge
    badge.style.setProperty('--badge-color', interpretation.color);

    if (settings.showOnHoverOnly) {
      badge.classList.add(`${BADGE_CLASS}--hover-only`);
    }

    wrapper.appendChild(badge);

    updateBadgeVisibility(badge);
  }

  function updateBadgeVisibility(badge) {
    if (settings.showOnHoverOnly) {
      badge.classList.add(`${BADGE_CLASS}--hover-only`);
    } else {
      badge.classList.remove(`${BADGE_CLASS}--hover-only`);
    }
  }

  function getLevelIcon(level) {
    switch (level) {
      case 'definite':  return '⬡'; // Hexagon — definite AI
      case 'likely':    return '◈'; // Diamond — likely AI
      case 'possible':  return '◇'; // Open diamond — possible
      case 'unlikely':  return '○'; // Circle — probably real
      case 'clean':     return '✓'; // Check — no signals
      default:          return '?';
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ---------------------------------------------------------------------------
  // Observation: initial scan + MutationObserver + IntersectionObserver
  // ---------------------------------------------------------------------------

  // Use IntersectionObserver to only process visible images (performance)
  const intersectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        intersectionObserver.unobserve(img);

        if (img.complete && img.naturalWidth > 0) {
          queueImage(img);
        } else {
          img.addEventListener('load', () => queueImage(img), { once: true });
        }
      }
    });
  }, {
    rootMargin: '200px', // Pre-load slightly off-screen
    threshold: 0,
  });

  function observeImage(img) {
    if (processedImages.has(img)) return;
    if (!img.src || img.src.startsWith('data:image/gif;base64,R0lGOD')) return; // 1px GIF
    intersectionObserver.observe(img);
  }

  // Initial scan
  function scanPage() {
    if (!settings.enabled) return;
    document.querySelectorAll('img[src]').forEach(observeImage);

    // Also handle <picture> elements
    document.querySelectorAll('picture img').forEach(observeImage);
  }

  // Watch for dynamically added images (SPAs, infinite scroll)
  const mutationObserver = new MutationObserver((mutations) => {
    if (!settings.enabled) return;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.tagName === 'IMG' && node.src) {
          observeImage(node);
        }
        // Check descendants
        node.querySelectorAll?.('img[src]').forEach(observeImage);
      }

      // Handle src attribute changes
      if (mutation.type === 'attributes' && mutation.target.tagName === 'IMG') {
        const img = mutation.target;
        processedImages.delete(img); // Allow re-analysis on src change
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

  // Kick off initial scan
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scanPage);
  } else {
    scanPage();
  }

  // Also scan after full page load (lazy-loaded images)
  window.addEventListener('load', () => {
    setTimeout(scanPage, 500);
  });

  // ---------------------------------------------------------------------------
  // Popup communication — expose page image list
  // ---------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'GET_PAGE_IMAGES') {
      const images = Array.from(document.querySelectorAll('img[src]'))
        .filter(img => img.src && !img.src.startsWith('chrome-extension://'))
        .map(img => ({
          src: img.src,
          width: img.naturalWidth,
          height: img.naturalHeight,
          alt: img.alt,
        }))
        .slice(0, 200); // Cap at 200

      sendResponse({ images });
      return false;
    }

    if (message.type === 'TOGGLE_ENABLED') {
      settings.enabled = message.enabled;
      if (!settings.enabled) {
        document.querySelectorAll(`.${WRAPPER_CLASS}`).forEach(wrapper => {
          wrapper.querySelectorAll(`.${BADGE_CLASS}`).forEach(b => b.style.display = 'none');
        });
      } else {
        document.querySelectorAll(`.${BADGE_CLASS}`).forEach(b => b.style.display = '');
        scanPage();
      }
      return false;
    }
  });

})();
