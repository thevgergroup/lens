#!/usr/bin/env node
/**
 * Quick diagnostic: measure noise residual signals on fixture images.
 * Run: node tests/fixtures/measure-noise-signals.mjs
 */

import sharp from 'sharp';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

// Inline the core logic from detector.js (avoids browser API deps)
function computeNoiseResidualStats(r, g, b, width, height, step) {
  const nR = [], nG = [], nB = [];
  for (let y = 1; y < height - 1; y += step) {
    for (let x = 1; x < width - 1; x += step) {
      const i = y * width + x;
      const neighbours = [
        i - width - 1, i - width, i - width + 1,
        i - 1,                    i + 1,
        i + width - 1, i + width, i + width + 1,
      ];
      const meanR = neighbours.reduce((s, j) => s + r[j], 0) / 8;
      const meanG = neighbours.reduce((s, j) => s + g[j], 0) / 8;
      const meanB = neighbours.reduce((s, j) => s + b[j], 0) / 8;
      nR.push(r[i] - meanR);
      nG.push(g[i] - meanG);
      nB.push(b[i] - meanB);
    }
  }

  const n = nR.length;
  if (n < 100) return { autocorr: 0, crossChannelCorr: 0 };

  const nLum = nR.map((v, i) => 0.299 * v + 0.587 * nG[i] + 0.114 * nB[i]);

  let cov1 = 0, varN = 0;
  const meanN = nLum.reduce((s, v) => s + v, 0) / n;
  for (let i = 0; i < n - 1; i++) {
    cov1 += (nLum[i] - meanN) * (nLum[i + 1] - meanN);
    varN  += (nLum[i] - meanN) ** 2;
  }
  varN += (nLum[n - 1] - meanN) ** 2;
  const autocorr = varN > 0 ? cov1 / varN : 0;

  let sumR = 0, sumB = 0;
  for (let i = 0; i < n; i++) { sumR += nR[i]; sumB += nB[i]; }
  const mR = sumR / n, mB = sumB / n;
  let covRB = 0, vR = 0, vB = 0;
  for (let i = 0; i < n; i++) {
    covRB += (nR[i] - mR) * (nB[i] - mB);
    vR    += (nR[i] - mR) ** 2;
    vB    += (nB[i] - mB) ** 2;
  }
  const crossChannelCorr = (vR > 0 && vB > 0) ? Math.abs(covRB / Math.sqrt(vR * vB)) : 0;
  return { autocorr: Math.max(0, autocorr), crossChannelCorr };
}

async function measureImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const img = sharp(fullPath).resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true });
    const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    const totalPixels = width * height;

    const r = new Float32Array(totalPixels);
    const g = new Float32Array(totalPixels);
    const b = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      const base = i * channels;
      r[i] = data[base];
      g[i] = data[base + 1];
      b[i] = data[base + 2];
    }

    const step = Math.max(1, Math.floor(Math.sqrt(totalPixels / 5000)));
    const stats = computeNoiseResidualStats(r, g, b, width, height, step);
    return { ...stats, width, height, totalPixels };
  } catch (err) {
    return { error: err.message };
  }
}

const AI_IMAGES = [
  'ai/flux-grid.jpg',
  'ai/flux-schnell-grid.jpg',
  'ai/sdxl-sample-01.jpg',
  'ai/sdxl-test.png',
  'ai/aurora-astronaut.webp',
  'ai/aurora-cyberpunk-city.webp',
  'ai/aurora-tea-dog.webp',
  'ai/aurora-mountain.webp',
  'ai/sd-txt2img-01.png',
  'ai/sd-img2img-mountains.png',
  'ai/sd-android.png',
  'ai/sd-golem.jpg',
  'ai/dalle-robot-letter.png',
  'ai/dalle-blue-man.png',
];

const REAL_IMAGES = [
  'real/crater-lake-nocreds.jpg',
  'real/ant-photo.jpg',
  'real/dog-photo.jpg',
  'real/flower-macro.jpg',
  'real/ocean-waves.jpg',
  'real/golden-gate-bridge.jpg',
  'real/waterfall.jpg',
  'real/cat-photo.jpg',
  'real/lena.png',
  'real/mountain-landscape.jpg',
  'real/picsum-landscape-10.jpg',
  'real/picsum-landscape-65.jpg',
  'real/nasa-astronaut-portrait.jpg',
];

console.log('Noise residual signal measurements on fixture images\n');
console.log('Thresholds: autocorr > 0.13 fires (0.22 = high), crossChan > 0.22 fires (0.38 = high)\n');

console.log('=== AI IMAGES ===');
console.log('image'.padEnd(40), 'autocorr'.padEnd(12), 'crossChan'.padEnd(12), 'fires?');
console.log('-'.repeat(80));
for (const img of AI_IMAGES) {
  const r = await measureImage(img);
  if (r.error) { console.log(img.padEnd(40), 'ERROR:', r.error); continue; }
  const autoFires = r.autocorr > 0.13 ? (r.autocorr > 0.22 ? 'HIGH' : 'low') : '';
  const crossFires = r.crossChannelCorr > 0.22 ? (r.crossChannelCorr > 0.38 ? 'HIGH' : 'low') : '';
  const fires = [autoFires && `auto:${autoFires}`, crossFires && `cross:${crossFires}`].filter(Boolean).join(' ');
  console.log(
    img.split('/').pop().padEnd(40),
    r.autocorr.toFixed(4).padEnd(12),
    r.crossChannelCorr.toFixed(4).padEnd(12),
    fires || 'MISS'
  );
}

console.log('\n=== REAL IMAGES ===');
console.log('image'.padEnd(40), 'autocorr'.padEnd(12), 'crossChan'.padEnd(12), 'false+?');
console.log('-'.repeat(80));
for (const img of REAL_IMAGES) {
  const r = await measureImage(img);
  if (r.error) { console.log(img.split('/').pop().padEnd(40), 'ERROR:', r.error); continue; }
  const autoFires = r.autocorr > 0.13 ? (r.autocorr > 0.22 ? 'HIGH!' : 'low') : '';
  const crossFires = r.crossChannelCorr > 0.22 ? (r.crossChannelCorr > 0.38 ? 'HIGH!' : 'low') : '';
  const fires = [autoFires && `auto:${autoFires}`, crossFires && `cross:${crossFires}`].filter(Boolean).join(' ');
  console.log(
    img.split('/').pop().padEnd(40),
    r.autocorr.toFixed(4).padEnd(12),
    r.crossChannelCorr.toFixed(4).padEnd(12),
    fires || 'clean'
  );
}
