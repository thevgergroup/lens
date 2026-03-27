#!/usr/bin/env node
/**
 * Implements the Mallet et al. (2025) noise correlation approach and measures
 * the raw feature values on our fixture set.
 *
 * Method:
 * 1. Convert RGB → YCbCr
 * 2. Apply Laplace filter (L4) per channel to extract noise residual
 * 3. Divide into 8×8 non-overlapping blocks
 * 4. Select T blocks with lowest variance (least content, most noise)
 * 5. Compute pairwise Pearson correlation between selected blocks
 * 6. Extract lower triangle of correlation matrix as feature vector
 *
 * Goal: understand the distribution of these features across AI vs real images
 * to determine if a fixed threshold or simple logistic regression suffices.
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

// RGB → YCbCr (BT.601)
function rgbToYCbCr(r, g, b) {
  const Y  =  0.299   * r + 0.587   * g + 0.114   * b;
  const Cb = -0.16874 * r - 0.33126 * g + 0.5     * b + 128;
  const Cr =  0.5     * r - 0.41869 * g - 0.08131 * b + 128;
  return [Y, Cb, Cr];
}

// 3×3 Laplace high-pass filter (L4 from paper)
// F[y,x] = -4*I[y,x] + I[y-1,x] + I[y+1,x] + I[y,x-1] + I[y,x+1]
function laplace(channel, width, height) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      out[i] = -4 * channel[i]
        + channel[i - width]
        + channel[i + width]
        + channel[i - 1]
        + channel[i + 1];
    }
  }
  return out;
}

// Extract non-overlapping 8×8 blocks from a filtered channel
// Returns array of Float32Array(64) vectors
function extractBlocks(filtered, width, height) {
  const blocks = [];
  for (let by = 0; by + 8 <= height; by += 8) {
    for (let bx = 0; bx + 8 <= width; bx += 8) {
      const block = new Float32Array(64);
      for (let dy = 0; dy < 8; dy++) {
        for (let dx = 0; dx < 8; dx++) {
          block[dy * 8 + dx] = filtered[(by + dy) * width + (bx + dx)];
        }
      }
      blocks.push(block);
    }
  }
  return blocks;
}

function blockVariance(block) {
  const n = block.length;
  let sum = 0;
  for (const v of block) sum += v;
  const mean = sum / n;
  let v = 0;
  for (const x of block) v += (x - mean) ** 2;
  return v / n;
}

function blockMean(block) {
  let s = 0;
  for (const v of block) s += Math.abs(v);
  return s / block.length;
}

// Select T blocks with lowest variance (most homogeneous = most noise)
function selectLowVarianceBlocks(blocks, T) {
  const scored = blocks.map((b, i) => ({ i, score: blockVariance(b) + blockMean(b) }));
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, T).map(s => blocks[s.i]);
}

// Pearson correlation between two block vectors
function pearson(a, b) {
  const n = a.length;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    cov += da * db; va += da * da; vb += db * db;
  }
  return (va > 0 && vb > 0) ? cov / Math.sqrt(va * vb) : 0;
}

// Build pairwise correlation matrix for selected blocks, return lower triangle
function buildCorrelationFeatures(blocks) {
  const N = blocks.length;
  const features = [];
  for (let i = 1; i < N; i++) {
    for (let j = 0; j < i; j++) {
      features.push(pearson(blocks[i], blocks[j]));
    }
  }
  return features;
}

// Summary statistics of a feature vector
function summarize(features) {
  if (features.length === 0) return { mean: 0, std: 0, absMax: 0, absMean: 0 };
  const n = features.length;
  let sum = 0, absSum = 0, absMax = 0;
  for (const v of features) {
    sum += v;
    absSum += Math.abs(v);
    absMax = Math.max(absMax, Math.abs(v));
  }
  const mean = sum / n;
  const absMean = absSum / n;
  let vsum = 0;
  for (const v of features) vsum += (v - mean) ** 2;
  return { mean, std: Math.sqrt(vsum / n), absMax, absMean };
}

async function analyzeImage(relPath, T = 20) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .toColorspace('srgb')
      .raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    if (channels < 3) throw new Error('Need RGB');

    // Convert to YCbCr channels
    const Yc  = new Float32Array(width * height);
    const Cbc = new Float32Array(width * height);
    const Crc = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const [Y, Cb, Cr] = rgbToYCbCr(data[i*channels], data[i*channels+1], data[i*channels+2]);
      Yc[i] = Y; Cbc[i] = Cb; Crc[i] = Cr;
    }

    // Apply Laplace per channel
    const fY  = laplace(Yc,  width, height);
    const fCb = laplace(Cbc, width, height);
    const fCr = laplace(Crc, width, height);

    // Extract 8×8 blocks
    const blocksY  = extractBlocks(fY,  width, height);
    const blocksCb = extractBlocks(fCb, width, height);
    const blocksCr = extractBlocks(fCr, width, height);

    // Select T lowest-variance blocks
    const selY  = selectLowVarianceBlocks(blocksY,  T);
    const selCb = selectLowVarianceBlocks(blocksCb, T);
    const selCr = selectLowVarianceBlocks(blocksCr, T);

    // Build correlation features per channel
    const featY  = buildCorrelationFeatures(selY);
    const featCb = buildCorrelationFeatures(selCb);
    const featCr = buildCorrelationFeatures(selCr);

    // Also compute joint (concatenate all channels)
    const featAll = [...featY, ...featCb, ...featCr];

    const sY  = summarize(featY);
    const sCb = summarize(featCb);
    const sCr = summarize(featCr);
    const sAll = summarize(featAll);

    return { sY, sCb, sCr, sAll, nBlocks: blocksY.length, nSelected: T };
  } catch (err) {
    return { error: err.message };
  }
}

const IMAGES = {
  'AI (no C2PA)': [
    'ai/flux-grid.jpg', 'ai/flux-schnell-grid.jpg',
    'ai/sdxl-sample-01.jpg', 'ai/sdxl-test.png',
    'ai/aurora-astronaut.webp', 'ai/aurora-cyberpunk-city.webp',
    'ai/aurora-tea-dog.webp', 'ai/aurora-mountain.webp', 'ai/aurora-vangogh-cat.webp',
    'ai/aurora-cherry-blossom.webp',
    'ai/sd-txt2img-01.png', 'ai/sd-img2img-mountains.png',
    'ai/sd-golem.jpg', 'ai/sd-android.png',
    'ai/dalle-blue-man.png', 'ai/dalle-robot-letter.png',
  ],
  'AI (C2PA)': [
    'ai/chatgpt-image.png', 'ai/firefly-tabby-cat.jpg',
    'ai/bing-creator-puppy.jpg', 'ai/dalle-puppy.webp',
  ],
  'Real': [
    'real/crater-lake-nocreds.jpg', 'real/ant-photo.jpg', 'real/dog-photo.jpg',
    'real/flower-macro.jpg', 'real/cloudscape.jpg', 'real/golden-gate-bridge.jpg',
    'real/cat-photo.jpg', 'real/mountain-landscape.jpg', 'real/lena.png',
    'real/nasa-astronaut-portrait.jpg', 'real/nasa-wildlife.jpg', 'real/nasa-madagascar-coast.jpg',
    'real/picsum-landscape-10.jpg', 'real/picsum-landscape-65.jpg',
    'real/picsum-portrait-91.jpg', 'real/waterfall.jpg', 'real/car-photo.jpg',
    'real/ocean-waves.jpg', 'real/city-nyc.jpg',
  ],
};

// Print header
console.log('Mallet et al. noise correlation features (T=20 lowest-variance blocks)\n');
console.log('Columns: absMean(Y), absMean(Cb), absMean(Cr), absMean(ALL), std(Y), std(Cb), std(Cr)');
console.log('Hypothesis: AI images → lower absMean correlation (more independent blocks)');
console.log('           Real images → higher absMean correlation (camera-correlated noise)\n');

for (const [group, images] of Object.entries(IMAGES)) {
  console.log(`=== ${group} ===`);
  console.log('image'.padEnd(35), 'amY'.padEnd(8), 'amCb'.padEnd(8), 'amCr'.padEnd(8), 'amALL'.padEnd(8), 'stdY'.padEnd(8), 'meanY');
  console.log('-'.repeat(90));
  for (const img of images) {
    const r = await analyzeImage(img);
    if (r.error) { console.log(img.split('/').pop().padEnd(35), 'ERROR:', r.error); continue; }
    console.log(
      img.split('/').pop().padEnd(35),
      r.sY.absMean.toFixed(4).padEnd(8),
      r.sCb.absMean.toFixed(4).padEnd(8),
      r.sCr.absMean.toFixed(4).padEnd(8),
      r.sAll.absMean.toFixed(4).padEnd(8),
      r.sY.std.toFixed(4).padEnd(8),
      r.sY.mean.toFixed(4)
    );
  }
  console.log();
}

// Find optimal threshold
console.log('\n--- Threshold search on absMean(ALL) ---');
const allData = {};
for (const [group, images] of Object.entries(IMAGES)) {
  for (const img of images) {
    const r = await analyzeImage(img);
    if (!r.error) allData[img] = { group, val: r.sAll.absMean };
  }
}

const vals = Object.values(allData).map(d => d.val).sort((a, b) => a - b);
const minV = vals[0], maxV = vals[vals.length - 1];
const step = (maxV - minV) / 100;

let bestThresh = 0, bestAcc = 0;
for (let t = minV; t <= maxV; t += step) {
  let correct = 0, total = 0;
  for (const { group, val } of Object.values(allData)) {
    const isAI = group.startsWith('AI');
    const predAI = val < t; // hypothesis: AI = lower correlation
    if (isAI === predAI) correct++;
    total++;
  }
  const acc = correct / total;
  if (acc > bestAcc) { bestAcc = acc; bestThresh = t; }
}

console.log(`Best threshold: ${bestThresh.toFixed(4)} → accuracy: ${(bestAcc*100).toFixed(1)}%`);
console.log('\nPer-image classification at best threshold:');
for (const [img, { group, val }] of Object.entries(allData)) {
  const isAI = group.startsWith('AI');
  const predAI = val < bestThresh;
  const correct = isAI === predAI;
  if (!correct) console.log(`  WRONG: ${img.split('/').pop().padEnd(35)} val=${val.toFixed(4)}  true=${isAI?'AI':'real'}  pred=${predAI?'AI':'real'}`);
}
