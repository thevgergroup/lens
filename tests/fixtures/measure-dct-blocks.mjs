#!/usr/bin/env node
/**
 * Measure DCT block statistics to find discriminating features between
 * AI images (Grok/Aurora) and real photos, both served via X's JPEG CDN.
 *
 * Key insight: JPEG compression creates 8×8 block artifacts in ALL images.
 * But AI generator decoders (VQ-VAE etc.) create *additional* structure on top.
 *
 * We measure:
 *  1. AC energy distribution across the 64 DCT coefficients
 *     AI images: energy concentrated in low-freq AC coefficients, very little in high-freq
 *     Real photos: more energy spread across mid/high AC coefficients (texture, detail)
 *
 *  2. Block variance uniformity
 *     AI images: flat regions are very flat (near-zero AC energy), edges are very sharp
 *     Real: more continuous distribution of block energies
 *
 *  3. Coefficient quantisation signature
 *     AI generators quantise differently than JPEG re-encoding
 *     Some coefficients show distinct rounding patterns
 *
 *  4. Inter-block AC coefficient correlation (key Mallet-style signal)
 *     Measure correlation of specific AC coefficients *between* neighbouring blocks
 *     in flat regions. AI images: lower variation (structured). Real: noisier.
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

function collectImages(dir, prefix) {
  const results = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const rel  = `${prefix}/${entry}`;
    if (statSync(full).isDirectory()) results.push(...collectImages(full, rel));
    else if (/\.(jpg|jpeg|png|webp)$/.test(entry)) results.push(rel);
  }
  return results;
}

// Approximate 1D DCT-II on 8 values (unnormalised)
function dct8(v) {
  const out = new Float32Array(8);
  for (let k = 0; k < 8; k++) {
    let s = 0;
    for (let n = 0; n < 8; n++) s += v[n] * Math.cos(Math.PI * k * (2*n+1) / 16);
    out[k] = s;
  }
  return out;
}

// 2D 8×8 DCT on a flat 64-element block
function dct2d8(blk) {
  const tmp = new Float32Array(64);
  // Row DCT
  for (let r = 0; r < 8; r++) {
    const row = blk.slice(r*8, r*8+8);
    const d = dct8(row);
    for (let c = 0; c < 8; c++) tmp[r*8+c] = d[c];
  }
  // Column DCT
  const out = new Float32Array(64);
  const col = new Float32Array(8);
  for (let c = 0; c < 8; c++) {
    for (let r = 0; r < 8; r++) col[r] = tmp[r*8+c];
    const d = dct8(col);
    for (let r = 0; r < 8; r++) out[r*8+c] = d[r];
  }
  return out;
}

async function measureDCT(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height } = info;
    const lum = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) lum[i] = data[i] - 128; // centre

    // Collect 8×8 block DCT coefficients
    // Separate flat blocks (low spatial variance) from textured blocks
    const flatAcEnergies  = []; // AC energy for flat 8×8 blocks
    const allAcHighFrac   = []; // fraction of energy in high-freq AC (coefficients 28–63)

    let totalBlocks = 0;
    let flatCount   = 0;

    for (let by = 0; by + 8 <= height; by += 8) {
      for (let bx = 0; bx + 8 <= width; bx += 8) {
        const blk = new Float32Array(64);
        let lumVar = 0, lumMean = 0;
        for (let r = 0; r < 8; r++)
          for (let c = 0; c < 8; c++) {
            const v = lum[(by+r)*width+(bx+c)];
            blk[r*8+c] = v;
            lumMean += v;
          }
        lumMean /= 64;
        for (const v of blk) lumVar += (v - lumMean) ** 2;
        lumVar /= 64;

        const dct = dct2d8(blk);
        const dc  = dct[0] ** 2;
        let acTotal = 0, acHigh = 0;
        for (let k = 1; k < 64; k++) {
          const e = dct[k] ** 2;
          acTotal += e;
          if (k >= 28) acHigh += e;
        }
        const highFrac = acTotal > 0 ? acHigh / acTotal : 0;

        totalBlocks++;
        if (lumVar < 50) { // flat block
          flatAcEnergies.push(acTotal);
          flatCount++;
        }
        allAcHighFrac.push(highFrac);
      }
    }

    if (flatAcEnergies.length < 5) return null;

    // Mean AC energy in flat blocks — AI: lower (smoother flat regions)
    const meanFlatAc = flatAcEnergies.reduce((a,b) => a+b, 0) / flatAcEnergies.length;

    // Coefficient of variation of flat AC energies — AI: lower (more uniform flatness)
    const stdFlatAc = Math.sqrt(
      flatAcEnergies.reduce((a,v) => a + (v - meanFlatAc)**2, 0) / flatAcEnergies.length
    );
    const cvFlatAc = meanFlatAc > 0 ? stdFlatAc / meanFlatAc : 0;

    // Mean fraction of energy in high-freq AC — AI: lower (smoother overall)
    const meanHighFrac = allAcHighFrac.reduce((a,b) => a+b, 0) / allAcHighFrac.length;

    // Flat fraction
    const flatFrac = flatCount / totalBlocks;

    return { meanFlatAc, cvFlatAc, meanHighFrac, flatFrac };
  } catch (e) {
    return null;
  }
}

const AI_IMAGES   = collectImages(join(IMAGES_DIR, 'ai'),   'ai');
const REAL_IMAGES = collectImages(join(IMAGES_DIR, 'real'), 'real');

function fmt(n, w=8) { return n == null ? 'N/A'.padStart(w) : n.toFixed(3).padStart(w); }

console.log('\nDCT block analysis\n');
console.log('meanFlatAc  = mean AC energy in flat 8×8 blocks     (AI: lower = smoother)');
console.log('cvFlatAc    = coeff of variation of flat AC energy   (AI: lower = more uniform)');
console.log('meanHighFrac= mean fraction of energy in AC[28-63]   (AI: lower = smoother)');
console.log('flatFrac    = fraction of 8×8 blocks that are flat\n');

const aiResults   = [];
const realResults = [];

console.log('=== AI IMAGES ===');
console.log('image'.padEnd(38) + ' mFlatAC  cvFlat highFrac flatFrac');
console.log('-'.repeat(78));
for (const img of AI_IMAGES.sort()) {
  const s = await measureDCT(img);
  if (!s) { console.log(img.split('/').pop().padEnd(38) + ' [skip]'); continue; }
  aiResults.push({ name: img.split('/').pop(), ...s });
  console.log(
    img.split('/').pop().padEnd(38),
    fmt(s.meanFlatAc), fmt(s.cvFlatAc), fmt(s.meanHighFrac), fmt(s.flatFrac)
  );
}

console.log('\n=== REAL IMAGES ===');
console.log('image'.padEnd(38) + ' mFlatAC  cvFlat highFrac flatFrac');
console.log('-'.repeat(78));
for (const img of REAL_IMAGES.sort()) {
  const s = await measureDCT(img);
  if (!s) { console.log(img.split('/').pop().padEnd(38) + ' [skip]'); continue; }
  realResults.push({ name: img.split('/').pop(), ...s });
  console.log(
    img.split('/').pop().padEnd(38),
    fmt(s.meanFlatAc), fmt(s.cvFlatAc), fmt(s.meanHighFrac), fmt(s.flatFrac)
  );
}

// Stats summary
function stats(arr, key) {
  const vals = arr.map(r => r[key]);
  const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
  const mn   = Math.min(...vals);
  const mx   = Math.max(...vals);
  return { mean, min: mn, max: mx };
}

console.log('\n--- Summary statistics ---');
for (const key of ['meanFlatAc', 'cvFlatAc', 'meanHighFrac']) {
  const ai   = stats(aiResults,   key);
  const real = stats(realResults, key);
  console.log(`\n${key}:`);
  console.log(`  AI:   mean=${ai.mean.toFixed(3)}  min=${ai.min.toFixed(3)}  max=${ai.max.toFixed(3)}`);
  console.log(`  Real: mean=${real.mean.toFixed(3)}  min=${real.min.toFixed(3)}  max=${real.max.toFixed(3)}`);
}

// Threshold search on meanHighFrac (most promising: AI should be lower)
console.log('\n--- Threshold search on meanHighFrac < threshold (AI = low high-freq energy) ---');
for (const t of [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]) {
  const tp = aiResults.filter(r => r.meanHighFrac < t).length;
  const fp = realResults.filter(r => r.meanHighFrac < t).length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec  = tp / aiResults.length;
  console.log(`  < ${t.toFixed(2)}: TP=${tp}/${aiResults.length} FP=${fp}/${realResults.length} prec=${(prec*100).toFixed(0)}% recall=${(rec*100).toFixed(0)}%`);
}

console.log('\n--- Threshold search on meanFlatAc < threshold (AI = smoother flat regions) ---');
for (const t of [500, 1000, 2000, 3000, 5000, 8000, 12000]) {
  const tp = aiResults.filter(r => r.meanFlatAc < t).length;
  const fp = realResults.filter(r => r.meanFlatAc < t).length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec  = tp / aiResults.length;
  console.log(`  < ${t}: TP=${tp}/${aiResults.length} FP=${fp}/${realResults.length} prec=${(prec*100).toFixed(0)}% recall=${(rec*100).toFixed(0)}%`);
}
