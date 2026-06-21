#!/usr/bin/env node
/**
 * Calibrate the Normalized Patch Residual (NPR) signal across fixture images.
 *
 * NPR: subtract each pixel from the mean of its 3×3 neighbourhood.
 * For AI images (VQ-VAE decoder): flat regions have very low residual variance,
 * edges have sharp residual spikes → high ratio between max and min block variance.
 * For real photos: sensor noise distributes residual variance more uniformly.
 *
 * Outputs per-image NPR stats to help set thresholds for detector.js.
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
    const rel = `${prefix}/${entry}`;
    if (statSync(full).isDirectory()) results.push(...collectImages(full, rel));
    else if (/\.(jpg|jpeg|png|webp)$/.test(entry)) results.push(rel);
  }
  return results;
}

/**
 * Compute NPR statistics for an image.
 *
 * Returns:
 *   nprRatio      – std(block_residual_var) / mean(block_residual_var) for flat blocks
 *                   High ratio = AI-like (very uniform flat regions + sharp edges)
 *   flatFrac      – fraction of 16×16 blocks considered "flat" (low luminance variance)
 *   meanFlatVar   – mean residual variance in flat blocks
 *   edgeMeanVar   – mean residual variance in edge blocks
 *   varRatio      – edgeMeanVar / meanFlatVar  (high = sharp edges relative to flat)
 */
async function measureNPR(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height } = info;
    const N = width * height;
    const lum = new Float32Array(N);
    for (let i = 0; i < N; i++) lum[i] = data[i];

    // 1. Compute NPR: pixel - mean(3×3 neighbourhood)
    const residual = new Float32Array(N);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        const mean = (
          lum[(y-1)*width+(x-1)] + lum[(y-1)*width+x] + lum[(y-1)*width+(x+1)] +
          lum[y    *width+(x-1)] + lum[i]              + lum[y    *width+(x+1)] +
          lum[(y+1)*width+(x-1)] + lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)]
        ) / 9;
        residual[i] = lum[i] - mean;
      }
    }

    // 2. Divide into 16×16 blocks, compute luminance variance and residual variance
    const BLOCK = 16;
    const flatBlocks = [];
    const edgeBlocks = [];

    for (let by = 1; by + BLOCK < height - 1; by += BLOCK) {
      for (let bx = 1; bx + BLOCK < width - 1; bx += BLOCK) {
        let lumSum = 0, lumSqSum = 0, resSum = 0, resSqSum = 0;
        const n = BLOCK * BLOCK;
        for (let dy = 0; dy < BLOCK; dy++) {
          for (let dx = 0; dx < BLOCK; dx++) {
            const idx = (by + dy) * width + (bx + dx);
            lumSum   += lum[idx];
            lumSqSum += lum[idx] * lum[idx];
            resSum   += residual[idx];
            resSqSum += residual[idx] * residual[idx];
          }
        }
        const lumVar = lumSqSum / n - (lumSum / n) ** 2;
        const resVar = resSqSum / n - (resSum / n) ** 2;
        if (lumVar < 50) flatBlocks.push(resVar);   // "flat" = low luminance variance
        else edgeBlocks.push(resVar);
      }
    }

    if (flatBlocks.length < 5) return null;

    const meanFlat = flatBlocks.reduce((a, b) => a + b, 0) / flatBlocks.length;
    const stdFlat  = Math.sqrt(
      flatBlocks.reduce((a, v) => a + (v - meanFlat) ** 2, 0) / flatBlocks.length
    );
    const nprRatio = meanFlat > 0.01 ? stdFlat / meanFlat : 0;

    const meanEdge = edgeBlocks.length > 0
      ? edgeBlocks.reduce((a, b) => a + b, 0) / edgeBlocks.length
      : 0;
    const varRatio = meanFlat > 0.01 ? meanEdge / meanFlat : 0;
    const flatFrac = flatBlocks.length / (flatBlocks.length + edgeBlocks.length);

    return { nprRatio, flatFrac, meanFlatVar: meanFlat, edgeMeanVar: meanEdge, varRatio };
  } catch (e) {
    return null;
  }
}

const AI_IMAGES   = collectImages(join(IMAGES_DIR, 'ai'),   'ai');
const REAL_IMAGES = collectImages(join(IMAGES_DIR, 'real'), 'real');

function fmt(n) { return n == null ? '   N/A' : n.toFixed(3).padStart(7); }

console.log('\nNPR calibration — columns: nprRatio | varRatio | flatFrac | meanFlatVar | edgeMeanVar');
console.log('nprRatio  = std(flat block residual variance) / mean  →  high = AI-like');
console.log('varRatio  = mean edge residual variance / mean flat   →  high = AI-like\n');

const aiResults   = [];
const realResults = [];

console.log('=== AI IMAGES ===');
console.log('image'.padEnd(38) + ' nprRatio varRatio flatFrac mFlatVar mEdgeVar');
console.log('-'.repeat(85));
for (const img of AI_IMAGES.sort()) {
  const s = await measureNPR(img);
  if (!s) { console.log(img.split('/').pop().padEnd(38) + ' [skip]'); continue; }
  aiResults.push({ name: img.split('/').pop(), ...s });
  console.log(
    img.split('/').pop().padEnd(38),
    fmt(s.nprRatio), fmt(s.varRatio), fmt(s.flatFrac),
    fmt(s.meanFlatVar), fmt(s.edgeMeanVar)
  );
}

console.log('\n=== REAL IMAGES ===');
console.log('image'.padEnd(38) + ' nprRatio varRatio flatFrac mFlatVar mEdgeVar');
console.log('-'.repeat(85));
for (const img of REAL_IMAGES.sort()) {
  const s = await measureNPR(img);
  if (!s) { console.log(img.split('/').pop().padEnd(38) + ' [skip]'); continue; }
  realResults.push({ name: img.split('/').pop(), ...s });
  console.log(
    img.split('/').pop().padEnd(38),
    fmt(s.nprRatio), fmt(s.varRatio), fmt(s.flatFrac),
    fmt(s.meanFlatVar), fmt(s.edgeMeanVar)
  );
}

// Threshold search on nprRatio
console.log('\n--- Threshold search on nprRatio ---');
for (const threshold of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]) {
  const tp = aiResults.filter(r => r.nprRatio > threshold).length;
  const fp = realResults.filter(r => r.nprRatio > threshold).length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = aiResults.length > 0 ? tp / aiResults.length : 0;
  console.log(
    `  nprRatio > ${threshold.toFixed(1)}: ` +
    `TP=${tp}/${aiResults.length} FP=${fp}/${realResults.length} ` +
    `prec=${(precision*100).toFixed(0)}% recall=${(recall*100).toFixed(0)}%`
  );
}

console.log('\n--- Threshold search on varRatio ---');
for (const threshold of [2, 3, 4, 5, 6, 8, 10, 15]) {
  const tp = aiResults.filter(r => r.varRatio > threshold).length;
  const fp = realResults.filter(r => r.varRatio > threshold).length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = aiResults.length > 0 ? tp / aiResults.length : 0;
  console.log(
    `  varRatio > ${threshold}: ` +
    `TP=${tp}/${aiResults.length} FP=${fp}/${realResults.length} ` +
    `prec=${(precision*100).toFixed(0)}% recall=${(recall*100).toFixed(0)}%`
  );
}
