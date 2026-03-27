#!/usr/bin/env node
/**
 * Tests the Synthbuster cross-difference periodicity approach.
 * D(x,y) = I(x,y) - I(x+1,y) - I(x,y+1) + I(x+1,y+1)
 *
 * This cross-difference filter is second-order; natural images have low spectrum
 * except near DC. LDM VAE upsampling creates periodic artifacts that appear as
 * peaks in the cross-difference spectrum at period P:
 *   SD 1.x:  P=4   (4x VAE)
 *   SDXL:    P=8   (8x VAE)
 *   FLUX.1:  P=16  (8x VAE × 2x patch)
 *   Aurora:  P=?   (VQGAN tokenizer, likely P=8 or P=16)
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

function computeCrossDiffPeaks(pixelData, width, height, channels) {
  const maxSize = 512;
  // Use luma
  const lum = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const b = i * channels;
    lum[i] = (0.299 * pixelData[b] + 0.587 * pixelData[b+1] + 0.114 * pixelData[b+2]) / 255;
  }

  // Cross-difference filter
  const D = new Float32Array((width-1) * (height-1));
  for (let y = 0; y < height-1; y++) {
    for (let x = 0; x < width-1; x++) {
      D[y*(width-1)+x] = lum[y*width+x] - lum[y*width+(x+1)] - lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)];
    }
  }

  // Compute 1D autocorrelation of D (row-wise) up to lag 32
  const dw = width - 1;
  const dh = height - 1;
  const maxLag = 32;
  const autocorr = new Float32Array(maxLag + 1);

  // Sample rows
  let count = 0;
  const step = Math.max(1, Math.floor(dh / 200));
  for (let y = 0; y < dh; y += step) {
    // Row mean
    let rowMean = 0;
    for (let x = 0; x < dw; x++) rowMean += D[y*dw+x];
    rowMean /= dw;

    let var0 = 0;
    for (let x = 0; x < dw; x++) var0 += (D[y*dw+x] - rowMean) ** 2;

    if (var0 < 1e-10) continue;

    for (let lag = 0; lag <= maxLag; lag++) {
      let cov = 0;
      for (let x = 0; x < dw - lag; x++) {
        cov += (D[y*dw+x] - rowMean) * (D[y*dw+x+lag] - rowMean);
      }
      autocorr[lag] += cov / var0;
    }
    count++;
  }

  if (count === 0) return {};
  for (let lag = 0; lag <= maxLag; lag++) autocorr[lag] /= count;

  // Find peaks at P=4,8,16 (skip P=multiples of 8 for JPEG? — check all for now)
  const periods = [4, 8, 16];
  const peaks = {};
  for (const P of periods) {
    peaks[`P${P}`] = autocorr[P];
  }
  peaks.baseline = (autocorr[3] + autocorr[5] + autocorr[7] + autocorr[9]) / 4;
  peaks.raw = Array.from(autocorr.slice(1, 20)).map(v => v.toFixed(3)).join(',');
  return peaks;
}

async function measureImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const img = sharp(fullPath).resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true });
    const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
    return computeCrossDiffPeaks(data, info.width, info.height, info.channels);
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
  'ai/sd-golem.jpg',
  'ai/dalle-robot-letter.png',
];

const REAL_IMAGES = [
  'real/crater-lake-nocreds.jpg',
  'real/ant-photo.jpg',
  'real/dog-photo.jpg',
  'real/flower-macro.jpg',
  'real/ocean-waves.jpg',
  'real/golden-gate-bridge.jpg',
  'real/cat-photo.jpg',
  'real/lena.png',
  'real/mountain-landscape.jpg',
  'real/picsum-landscape-10.jpg',
];

console.log('Cross-difference periodicity analysis\n');
console.log('P4=4-period peak, P8=8-period peak, P16=16-period peak, baseline=avg(non-period lags)\n');

console.log('=== AI IMAGES ===');
console.log('image'.padEnd(35), 'P4'.padEnd(8), 'P8'.padEnd(8), 'P16'.padEnd(8), 'baseline', '  lags(1-19)');
console.log('-'.repeat(110));
for (const img of AI_IMAGES) {
  const r = await measureImage(img);
  if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
  console.log(
    img.split('/').pop().padEnd(35),
    (r.P4 || 0).toFixed(4).padEnd(8),
    (r.P8 || 0).toFixed(4).padEnd(8),
    (r.P16 || 0).toFixed(4).padEnd(8),
    (r.baseline || 0).toFixed(4).padEnd(10),
    r.raw
  );
}

console.log('\n=== REAL IMAGES ===');
console.log('image'.padEnd(35), 'P4'.padEnd(8), 'P8'.padEnd(8), 'P16'.padEnd(8), 'baseline', '  lags(1-19)');
console.log('-'.repeat(110));
for (const img of REAL_IMAGES) {
  const r = await measureImage(img);
  if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
  console.log(
    img.split('/').pop().padEnd(35),
    (r.P4 || 0).toFixed(4).padEnd(8),
    (r.P8 || 0).toFixed(4).padEnd(8),
    (r.P16 || 0).toFixed(4).padEnd(8),
    (r.baseline || 0).toFixed(4).padEnd(10),
    r.raw
  );
}
