#!/usr/bin/env node
/**
 * Tests Benford's Law on DCT coefficients as an AI image detector.
 *
 * Method (arxiv:2004.07682):
 * 1. Compute 8x8 block DCT of luma channel
 * 2. Extract first significant digit of each non-DC coefficient
 * 3. Chi-squared test against Benford distribution
 * 4. Also test: spectral flatness (Wiener entropy) of magnitude spectrum
 *    and ratio of high-frequency to low-frequency energy
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

// Precomputed 8x8 DCT-II basis (cos((2i+1)*j*pi/16) for i=0..7, j=0..7)
function buildDctMatrix() {
  const C = [];
  for (let j = 0; j < 8; j++) {
    C[j] = [];
    for (let i = 0; i < 8; i++) {
      C[j][i] = Math.cos((2*i+1)*j*Math.PI/16);
    }
  }
  return C;
}
const DCT_MATRIX = buildDctMatrix();

// 8x8 DCT-II: output[u][v] = sum_{x,y} block[x][y] * cos(...x...) * cos(...y...)
function dct8x8(block) {
  const out = Array.from({length: 8}, () => new Float32Array(8));
  for (let u = 0; u < 8; u++) {
    for (let v = 0; v < 8; v++) {
      let sum = 0;
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          sum += block[x][y] * DCT_MATRIX[u][x] * DCT_MATRIX[v][y];
        }
      }
      const cu = u === 0 ? 1/Math.SQRT2 : 1;
      const cv = v === 0 ? 1/Math.SQRT2 : 1;
      out[u][v] = 0.25 * cu * cv * sum;
    }
  }
  return out;
}

function firstSignificantDigit(v) {
  const abs = Math.abs(v);
  if (abs < 1e-10) return null;
  const s = abs.toExponential();
  for (const c of s) {
    if (c >= '1' && c <= '9') return parseInt(c);
  }
  return null;
}

// Benford's expected distribution for digits 1-9
const BENFORD = [0, Math.log10(2), Math.log10(1.5), Math.log10(4/3), Math.log10(5/4),
                 Math.log10(6/5), Math.log10(7/6), Math.log10(8/7), Math.log10(9/8), Math.log10(10/9)];
// Index 1-9: probability that first digit = d
const BENFORD_PROB = BENFORD.map((v, i) => i === 0 ? 0 : v);

function chiSquaredBenford(digitCounts, total) {
  let chi2 = 0;
  for (let d = 1; d <= 9; d++) {
    const expected = BENFORD_PROB[d] * total;
    if (expected < 1) continue;
    const diff = digitCounts[d] - expected;
    chi2 += (diff * diff) / expected;
  }
  return chi2;
}

// Spectral flatness (Wiener entropy): geometric mean / arithmetic mean of power spectrum
// High flatness = noise-like (real camera), low flatness = tonal/structured (AI)
function spectralFlatness(magnitudes) {
  const n = magnitudes.length;
  let logSum = 0, sum = 0;
  let nonzero = 0;
  for (const m of magnitudes) {
    if (m > 1e-10) {
      logSum += Math.log(m * m); // log of power
      sum += m * m;
      nonzero++;
    }
  }
  if (nonzero === 0 || sum === 0) return 0;
  const geoMean = Math.exp(logSum / nonzero);
  const ariMean = sum / nonzero;
  return geoMean / ariMean; // 0 = tonal, 1 = flat/noisy
}

// HF/LF energy ratio in the 2D FFT magnitude spectrum
function hfLfRatio(mag, size) {
  const mid = size / 2;
  const radius = size / 4; // boundary between LF and HF regions
  let hf = 0, lf = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      // Distance from DC (shifted to center)
      const dy = y < mid ? y : y - size;
      const dx = x < mid ? x : x - size;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const m2 = mag[y * size + x] ** 2;
      if (dist < radius) lf += m2;
      else hf += m2;
    }
  }
  return lf > 0 ? hf / lf : 0;
}

// Simple 1D FFT (Cooley-Tukey)
function fft1d(re, im) {
  const N = re.length;
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
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
        re[i+k] = ur+vr; im[i+k] = ui+vi;
        re[i+k+len/2] = ur-vr; im[i+k+len/2] = ui-vi;
        const tr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = tr;
      }
    }
  }
}

function largestPow2Leq(n) { let p=1; while(p*2<=n) p*=2; return p; }

async function analyzeImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .greyscale()
      .raw().toBuffer({ resolveWithObject: true });
    const { width, height } = info;

    // --- Benford's Law on 8x8 DCT blocks ---
    const digitCounts = new Array(10).fill(0);
    let totalCoeffs = 0;
    const block = Array.from({length: 8}, () => new Array(8));

    for (let by = 0; by + 8 <= height; by += 8) {
      for (let bx = 0; bx + 8 <= width; bx += 8) {
        for (let y = 0; y < 8; y++)
          for (let x = 0; x < 8; x++)
            block[y][x] = data[(by+y)*width+(bx+x)] - 128; // center

        const dct = dct8x8(block);
        for (let u = 0; u < 8; u++) {
          for (let v = 0; v < 8; v++) {
            if (u === 0 && v === 0) continue; // skip DC
            const d = firstSignificantDigit(dct[u][v]);
            if (d !== null) { digitCounts[d]++; totalCoeffs++; }
          }
        }
      }
    }
    const chi2 = chiSquaredBenford(digitCounts, totalCoeffs);
    // Critical value at p=0.001, 8 df = 26.12; p=0.05 = 15.51
    const benfordFires = chi2 > 26;

    // --- 2D FFT spectral analysis ---
    const fftSize = largestPow2Leq(Math.min(width, height, 256));
    const N = fftSize * fftSize;
    const re = new Float32Array(N);
    const im = new Float32Array(N);

    for (let y = 0; y < fftSize; y++)
      for (let x = 0; x < fftSize; x++)
        re[y*fftSize+x] = data[y*width+x] / 255;

    // Row FFTs
    for (let y = 0; y < fftSize; y++) {
      const rr = Float32Array.from(re.slice(y*fftSize, (y+1)*fftSize));
      const ri = new Float32Array(fftSize);
      fft1d(rr, ri);
      re.set(rr, y*fftSize); im.set(ri, y*fftSize);
    }
    // Col FFTs
    const cr = new Float32Array(fftSize), ci = new Float32Array(fftSize);
    for (let x = 0; x < fftSize; x++) {
      for (let y = 0; y < fftSize; y++) { cr[y] = re[y*fftSize+x]; ci[y] = im[y*fftSize+x]; }
      fft1d(cr, ci);
      for (let y = 0; y < fftSize; y++) { re[y*fftSize+x] = cr[y]; im[y*fftSize+x] = ci[y]; }
    }

    const mag = new Float32Array(N);
    for (let i = 0; i < N; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);

    // Skip DC (which dominates)
    const magNodc = Array.from(mag);
    magNodc[0] = 0;

    const flatness = spectralFlatness(magNodc);
    const hflf = hfLfRatio(mag, fftSize);

    return { chi2: chi2.toFixed(1), benfordFires, flatness: flatness.toFixed(4), hflf: hflf.toFixed(4), totalCoeffs };
  } catch(err) {
    return { error: err.message };
  }
}

const IMAGES = {
  'AI (no C2PA)': [
    'ai/flux-grid.jpg', 'ai/flux-schnell-grid.jpg',
    'ai/sdxl-sample-01.jpg', 'ai/sdxl-test.png',
    'ai/aurora-astronaut.webp', 'ai/aurora-cyberpunk-city.webp',
    'ai/aurora-tea-dog.webp', 'ai/aurora-mountain.webp', 'ai/aurora-vangogh-cat.webp',
    'ai/sd-txt2img-01.png', 'ai/sd-img2img-mountains.png',
    'ai/sd-golem.jpg', 'ai/dalle-blue-man.png', 'ai/dalle-robot-letter.png',
    'ai/sd-android.png',
  ],
  'AI (C2PA)': [
    'ai/chatgpt-image.png', 'ai/firefly-tabby-cat.jpg',
    'ai/bing-creator-puppy.jpg', 'ai/dalle-puppy.webp',
  ],
  'Real': [
    'real/crater-lake-nocreds.jpg', 'real/ant-photo.jpg', 'real/dog-photo.jpg',
    'real/flower-macro.jpg', 'real/ocean-waves.jpg', 'real/golden-gate-bridge.jpg',
    'real/cat-photo.jpg', 'real/lena.png', 'real/mountain-landscape.jpg',
    'real/cloudscape.jpg', 'real/picsum-landscape-10.jpg', 'real/picsum-landscape-65.jpg',
    'real/nasa-astronaut-portrait.jpg', 'real/nasa-wildlife.jpg',
    'real/waterfall.jpg', 'real/car-photo.jpg',
  ],
};

console.log('Benford DCT + Spectral analysis\n');
console.log('chi2 = Benford chi-squared (fires if > 26), flatness = spectral flatness (0=tonal, 1=noisy), hflf = HF/LF ratio\n');

for (const [group, images] of Object.entries(IMAGES)) {
  console.log(`=== ${group} ===`);
  console.log('image'.padEnd(35), 'chi2'.padEnd(10), 'B?'.padEnd(5), 'flatness'.padEnd(12), 'hf/lf');
  console.log('-'.repeat(75));
  for (const img of images) {
    const r = await analyzeImage(img);
    if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
    console.log(
      img.split('/').pop().padEnd(35),
      r.chi2.padEnd(10),
      (r.benfordFires ? 'YES' : 'no').padEnd(5),
      r.flatness.padEnd(12),
      r.hflf
    );
  }
  console.log();
}
