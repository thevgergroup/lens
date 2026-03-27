#!/usr/bin/env node
/**
 * Measures VAE grid artifact strength in the 2D DFT of the cross-difference image.
 *
 * The cross-difference filter D(x,y) = I(x,y) - I(x+1,y) - I(x,y+1) + I(x+1,y+1)
 * is a 2nd-order finite difference that amplifies periodic upsampling artifacts.
 *
 * For LDM VAE with P=8 upsampling, peaks appear at DFT bin k = N/P (and harmonics).
 * We measure the ratio of power at periodic bins vs average power.
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

// Minimal 1D FFT (Cooley-Tukey, power of 2)
function fft1d(re, im) {
  const N = re.length;
  // Bit-reversal
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len;
    for (let i = 0; i < N; i += len) {
      let cr = 1, ci = 0;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let k = 0; k < len >> 1; k++) {
        const ur = re[i+k], ui = im[i+k];
        const vr = re[i+k+len/2]*cr - im[i+k+len/2]*ci;
        const vi = re[i+k+len/2]*ci + im[i+k+len/2]*cr;
        re[i+k] = ur + vr; im[i+k] = ui + vi;
        re[i+k+len/2] = ur - vr; im[i+k+len/2] = ui - vi;
        const tr = cr*wr - ci*wi;
        ci = cr*wi + ci*wr; cr = tr;
      }
    }
  }
}

function largestPow2Leq(n) {
  let p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

function measureVaePeriodicity(pixelData, width, height, channels) {
  // Extract luma
  const lum = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const b = i * channels;
    lum[i] = (0.299 * pixelData[b] + 0.587 * pixelData[b+1] + 0.114 * pixelData[b+2]) / 255;
  }

  // Cross-difference filter
  const dw = width - 1, dh = height - 1;
  const D = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      D[y*dw+x] = lum[y*width+x] - lum[y*width+(x+1)] - lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)];
    }
  }

  // 2D FFT: use pow2 square crop
  const fftSize = largestPow2Leq(Math.min(dw, dh, 512));
  const N = fftSize * fftSize;
  const re = new Float32Array(N);
  const im = new Float32Array(N);

  // Copy D into re (top-left crop)
  for (let y = 0; y < fftSize; y++) {
    for (let x = 0; x < fftSize; x++) {
      re[y * fftSize + x] = D[y * dw + x];
    }
  }

  // Row-wise FFT
  const rowRe = new Float32Array(fftSize);
  const rowIm = new Float32Array(fftSize);
  for (let y = 0; y < fftSize; y++) {
    for (let x = 0; x < fftSize; x++) { rowRe[x] = re[y*fftSize+x]; rowIm[x] = 0; }
    fft1d(rowRe, rowIm);
    for (let x = 0; x < fftSize; x++) { re[y*fftSize+x] = rowRe[x]; im[y*fftSize+x] = rowIm[x]; }
  }

  // Column-wise FFT
  const colRe = new Float32Array(fftSize);
  const colIm = new Float32Array(fftSize);
  for (let x = 0; x < fftSize; x++) {
    for (let y = 0; y < fftSize; y++) { colRe[y] = re[y*fftSize+x]; colIm[y] = im[y*fftSize+x]; }
    fft1d(colRe, colIm);
    for (let y = 0; y < fftSize; y++) { re[y*fftSize+x] = colRe[y]; im[y*fftSize+x] = colIm[y]; }
  }

  // Compute magnitude spectrum
  const mag = new Float32Array(N);
  for (let i = 0; i < N; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);

  // For period P, the grid peaks are at DFT bins (k*fftSize/P, l*fftSize/P) for k,l != 0
  // We measure the average magnitude at grid positions vs. average of all non-DC magnitudes
  const results = {};

  for (const P of [4, 8, 16]) {
    const step = fftSize / P;
    if (!Number.isInteger(step) || step < 1) continue;

    let gridPower = 0, gridCount = 0;
    let totalPower = 0, totalCount = 0;

    for (let ky = 0; ky < fftSize; ky++) {
      for (let kx = 0; kx < fftSize; kx++) {
        if (ky === 0 && kx === 0) continue; // skip DC
        const m = mag[ky * fftSize + kx];
        totalPower += m;
        totalCount++;
        if (ky % step === 0 && kx % step === 0) {
          gridPower += m;
          gridCount++;
        }
      }
    }

    const avgGrid = gridCount > 0 ? gridPower / gridCount : 0;
    const avgAll = totalCount > 0 ? totalPower / totalCount : 0;
    results[`P${P}_ratio`] = avgAll > 0 ? avgGrid / avgAll : 0;
  }

  return results;
}

async function measureImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .raw().toBuffer({ resolveWithObject: true });
    return measureVaePeriodicity(data, info.width, info.height, info.channels);
  } catch (err) {
    return { error: err.message };
  }
}

const IMAGES = {
  'AI': [
    'ai/flux-grid.jpg', 'ai/flux-schnell-grid.jpg',
    'ai/sdxl-sample-01.jpg', 'ai/sdxl-test.png',
    'ai/aurora-astronaut.webp', 'ai/aurora-cyberpunk-city.webp', 'ai/aurora-mountain.webp',
    'ai/sd-txt2img-01.png', 'ai/sd-img2img-mountains.png',
    'ai/sd-golem.jpg', 'ai/dalle-robot-letter.png',
  ],
  'Real': [
    'real/crater-lake-nocreds.jpg', 'real/ant-photo.jpg', 'real/dog-photo.jpg',
    'real/flower-macro.jpg', 'real/ocean-waves.jpg', 'real/golden-gate-bridge.jpg',
    'real/cat-photo.jpg', 'real/lena.png', 'real/mountain-landscape.jpg',
    'real/cloudscape.jpg', 'real/picsum-landscape-10.jpg', 'real/nasa-astronaut-portrait.jpg',
  ],
};

console.log('VAE upsampling grid artifact measurement (2D DFT of cross-difference)\n');
console.log('ratio = avg power at grid bins / avg power at all non-DC bins');
console.log('If VAE artifacts present, ratio >> 1 at period P\n');

for (const [group, images] of Object.entries(IMAGES)) {
  console.log(`=== ${group.toUpperCase()} ===`);
  console.log('image'.padEnd(35), 'P4_ratio'.padEnd(12), 'P8_ratio'.padEnd(12), 'P16_ratio');
  console.log('-'.repeat(75));
  for (const img of images) {
    const r = await measureImage(img);
    if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
    console.log(
      img.split('/').pop().padEnd(35),
      (r.P4_ratio || 0).toFixed(4).padEnd(12),
      (r.P8_ratio || 0).toFixed(4).padEnd(12),
      (r.P16_ratio || 0).toFixed(4)
    );
  }
  console.log();
}
