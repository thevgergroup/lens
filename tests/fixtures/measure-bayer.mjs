#!/usr/bin/env node
/**
 * Tests camera-physics signals from arXiv:2310.16684 (Wong & Ng, ICCV 2024):
 *
 * Real cameras use Bayer pattern demosaicing:
 *   - Green pixels sampled at 2x rate of Red/Blue (RGGB pattern)
 *   - Interpolation creates specific correlation patterns between channels
 *   - Diagonal gradients show peaks at 0.5 normalized frequency in DFT
 *
 * AI images are generated without a camera sensor:
 *   - No Bayer pattern → no interpolation artifacts
 *   - RGB channels generated independently (or from latent)
 *   - Block variances more correlated across R/G/B
 *
 * Feature 1: DFT of diagonal gradients — look for peak at freq=0.5
 * Feature 2: Block-reduced diagonal variances (10x10 blocks)
 * Feature 3: Local Pearson correlation of block variances across channels
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

// 1D FFT
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

function pearsonCorr(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa/n, mb = sb/n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    cov += (a[i]-ma)*(b[i]-mb);
    va += (a[i]-ma)**2;
    vb += (b[i]-mb)**2;
  }
  return (va > 0 && vb > 0) ? cov / Math.sqrt(va * vb) : 0;
}

async function analyzeImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    // Get raw RGB (not greyscale — need per-channel)
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .toColorspace('srgb')
      .raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;

    if (channels < 3) throw new Error('Need RGB');

    const R = new Float32Array(width * height);
    const G = new Float32Array(width * height);
    const B = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      R[i] = data[i * channels];
      G[i] = data[i * channels + 1];
      B[i] = data[i * channels + 2];
    }

    // ── Feature 1: Bayer demosaicing artifact at 0.5 frequency ─────────────
    // Compute diagonal gradient: D[y,x] = (R[y,x] - R[y+1,x+1]) for one diagonal
    // Real cameras: green channel sampled at twice the rate → DFT shows peak at freq=N/2
    //
    // Method: compute mean absolute diagonal gradient per row, then 1D FFT
    // and measure power at the Nyquist bin (freq = 0.5) relative to neighbours

    function bayerFreqScore(channel) {
      // Per-row mean of diagonal differences
      const rowMeans = [];
      for (let y = 0; y < height - 1; y++) {
        let sum = 0;
        for (let x = 0; x < width - 1; x++) {
          sum += Math.abs(channel[y*width+x] - channel[(y+1)*width+(x+1)]);
        }
        rowMeans.push(sum / (width - 1));
      }

      // 1D FFT of row means
      const N2 = largestPow2Leq(rowMeans.length);
      const re = new Float32Array(N2);
      const im = new Float32Array(N2);
      for (let i = 0; i < N2; i++) re[i] = rowMeans[i] || 0;
      fft1d(re, im);

      // Magnitude
      const mag = new Float32Array(N2);
      for (let i = 0; i < N2; i++) mag[i] = Math.sqrt(re[i]*re[i] + im[i]*im[i]);

      // Nyquist bin is N2/2, check ratio vs neighbours
      const nyquistBin = N2 / 2;
      const nyquistMag = mag[nyquistBin];
      // Average of surrounding bins (excluding DC and Nyquist)
      let surroundSum = 0, count = 0;
      for (let k = 1; k < N2/2; k++) {
        if (Math.abs(k - nyquistBin) > N2/8) { surroundSum += mag[k]; count++; }
      }
      const surroundAvg = count > 0 ? surroundSum / count : 1;
      return nyquistMag / (surroundAvg + 1e-10);
    }

    const bayerR = bayerFreqScore(R);
    const bayerG = bayerFreqScore(G);
    const bayerB = bayerFreqScore(B);
    // G should have stronger Bayer artifact than R and B in real photos
    const bayerGoverRB = bayerG / (0.5 * bayerR + 0.5 * bayerB + 1e-10);

    // ── Feature 2 & 3: Block diagonal variance and cross-channel correlation ─
    // For each 10x10 block, compute variance of diagonal elements
    const blockSize = 10;
    const bw = Math.floor(width / blockSize);
    const bh = Math.floor(height / blockSize);

    const blockVarR = new Float32Array(bw * bh);
    const blockVarG = new Float32Array(bw * bh);
    const blockVarB = new Float32Array(bw * bh);

    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        const idx = by * bw + bx;
        // Diagonal variance within this block
        for (const [ch, dest] of [[R, blockVarR], [G, blockVarG], [B, blockVarB]]) {
          const diag = [], antidiag = [];
          for (let k = 0; k < blockSize - 1; k++) {
            const y = by*blockSize + k, x = bx*blockSize + k;
            if (y < height && x < width && y+1 < height && x+1 < width) {
              diag.push(ch[y*width+x] - ch[(y+1)*width+(x+1)]);
              antidiag.push(ch[y*width+(x+1)] - ch[(y+1)*width+x]);
            }
          }
          let varD = 0, varA = 0;
          const mD = diag.reduce((s,v)=>s+v,0)/diag.length;
          const mA = antidiag.reduce((s,v)=>s+v,0)/antidiag.length;
          for (const v of diag) varD += (v-mD)**2;
          for (const v of antidiag) varA += (v-mA)**2;
          dest[idx] = varD + varA;
        }
      }
    }

    // Cross-channel Pearson correlation of block variances
    const corrRG = pearsonCorr(blockVarR, blockVarG);
    const corrGB = pearsonCorr(blockVarG, blockVarB);
    const corrRB = pearsonCorr(blockVarR, blockVarB);
    const avgCrossCorr = (corrRG + corrGB + corrRB) / 3;

    // Also: R/G variance ratio (real cameras: G variance >> R,B due to 2x sampling)
    const sumR = Array.from(blockVarR).reduce((s,v)=>s+v,0);
    const sumG = Array.from(blockVarG).reduce((s,v)=>s+v,0);
    const sumB = Array.from(blockVarB).reduce((s,v)=>s+v,0);
    const gOverRB = sumG / (0.5*(sumR + sumB) + 1e-10);

    return {
      bayerR: bayerR.toFixed(3),
      bayerG: bayerG.toFixed(3),
      bayerB: bayerB.toFixed(3),
      bayerGoverRB: bayerGoverRB.toFixed(3),
      corrRG: corrRG.toFixed(3),
      corrGB: corrGB.toFixed(3),
      avgCrossCorr: avgCrossCorr.toFixed(3),
      gOverRB: gOverRB.toFixed(3),
    };
  } catch(err) {
    return { error: err.message };
  }
}

const IMAGES = {
  'AI (no C2PA) — should show NO Bayer pattern': [
    'ai/flux-grid.jpg', 'ai/flux-schnell-grid.jpg',
    'ai/sdxl-sample-01.jpg', 'ai/sdxl-test.png',
    'ai/aurora-astronaut.webp', 'ai/aurora-cyberpunk-city.webp',
    'ai/aurora-tea-dog.webp', 'ai/aurora-mountain.webp',
    'ai/sd-txt2img-01.png', 'ai/sd-golem.jpg',
  ],
  'Real camera photos — should show Bayer pattern': [
    'real/crater-lake-nocreds.jpg', 'real/ant-photo.jpg', 'real/dog-photo.jpg',
    'real/flower-macro.jpg', 'real/cloudscape.jpg', 'real/golden-gate-bridge.jpg',
    'real/cat-photo.jpg', 'real/mountain-landscape.jpg',
    'real/nasa-astronaut-portrait.jpg', 'real/nasa-wildlife.jpg',
    'real/waterfall.jpg', 'real/car-photo.jpg',
  ],
  'Synthetic/computer-rendered real images (no camera)': [
    'real/lena.png',
    'ai/dalle-robot-letter.png', 'ai/dalle-blue-man.png',
  ],
};

console.log('Bayer pattern + block variance cross-channel correlation\n');
console.log('bayerGoverRB: G diagonal freq > R/B? (expect >1 for real camera)');
console.log('avgCrossCorr: block variance correlation across R/G/B (expect <0.5 for real camera)\n');

for (const [group, images] of Object.entries(IMAGES)) {
  console.log(`=== ${group} ===`);
  console.log('image'.padEnd(35), 'bR'.padEnd(7), 'bG'.padEnd(7), 'bB'.padEnd(7), 'G/RB'.padEnd(8), 'corrRG'.padEnd(8), 'corrGB'.padEnd(8), 'avgCC');
  console.log('-'.repeat(90));
  for (const img of images) {
    const r = await analyzeImage(img);
    if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
    console.log(
      img.split('/').pop().padEnd(35),
      r.bayerR.padEnd(7), r.bayerG.padEnd(7), r.bayerB.padEnd(7),
      r.bayerGoverRB.padEnd(8), r.corrRG.padEnd(8), r.corrGB.padEnd(8),
      r.avgCrossCorr
    );
  }
  console.log();
}
