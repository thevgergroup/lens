#!/usr/bin/env node
/**
 * Deeper analysis of the noise correlation feature distributions.
 * Tries multiple T values, different summary statistics, and combinations
 * to find if any single derived scalar separates AI from real images.
 *
 * Also tests the correct direction from the paper: the paper trains a
 * logistic regression on the full lower-triangle vector. We test whether
 * the variance (spread) of the correlation values, or the proportion of
 * high-correlation pairs, separates the classes.
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

function rgbToYCbCr(r, g, b) {
  return [
     0.299   * r + 0.587   * g + 0.114   * b,
    -0.16874 * r - 0.33126 * g + 0.5     * b + 128,
     0.5     * r - 0.41869 * g - 0.08131 * b + 128,
  ];
}

function laplace(channel, width, height) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      out[i] = -4*channel[i] + channel[i-width] + channel[i+width] + channel[i-1] + channel[i+1];
    }
  }
  return out;
}

function extractBlocks(filtered, width, height) {
  const blocks = [];
  for (let by = 0; by + 8 <= height; by += 8) {
    for (let bx = 0; bx + 8 <= width; bx += 8) {
      const block = new Float32Array(64);
      for (let dy = 0; dy < 8; dy++)
        for (let dx = 0; dx < 8; dx++)
          block[dy*8+dx] = filtered[(by+dy)*width+(bx+dx)];
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
  for (const x of block) v += (x-mean)**2;
  return v / n;
}

function selectLowVarianceBlocks(blocks, T) {
  const scored = blocks.map((b, i) => ({ i, v: blockVariance(b) }));
  scored.sort((a, b) => a.v - b.v);
  // Skip near-zero variance blocks (all-zero = undefined correlation)
  const nonzero = scored.filter(s => s.v > 0.01);
  return nonzero.slice(0, T).map(s => blocks[s.i]);
}

function pearson(a, b) {
  const n = a.length;
  let sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa/n, mb = sb/n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i]-ma, db = b[i]-mb;
    cov += da*db; va += da*da; vb += db*db;
  }
  return (va > 1e-10 && vb > 1e-10) ? cov/Math.sqrt(va*vb) : 0;
}

function correlationStats(blocks) {
  if (blocks.length < 4) return null;
  const pairs = [];
  for (let i = 1; i < blocks.length; i++)
    for (let j = 0; j < i; j++)
      pairs.push(pearson(blocks[i], blocks[j]));

  const n = pairs.length;
  let sum = 0, absSum = 0;
  for (const v of pairs) { sum += v; absSum += Math.abs(v); }
  const mean = sum / n;
  const absMean = absSum / n;

  let vsum = 0, skewSum = 0;
  for (const v of pairs) {
    vsum += (v-mean)**2;
  }
  const std = Math.sqrt(vsum/n);

  // Skewness
  for (const v of pairs) skewSum += ((v-mean)/Math.max(std,1e-10))**3;
  const skew = skewSum / n;

  // Proportion of |r| > 0.3 (strongly correlated pairs)
  const highCorrFrac = pairs.filter(v => Math.abs(v) > 0.3).length / n;

  // Proportion of |r| > 0.15
  const modCorrFrac = pairs.filter(v => Math.abs(v) > 0.15).length / n;

  return { mean, absMean, std, skew, highCorrFrac, modCorrFrac, n };
}

async function analyzeImage(relPath, T = 30) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .toColorspace('srgb')
      .removeAlpha()
      .raw().toBuffer({ resolveWithObject: true });
    const { width, height } = info;
    const channels = 3;

    const Yc = new Float32Array(width*height);
    const Cbc = new Float32Array(width*height);
    const Crc = new Float32Array(width*height);
    for (let i = 0; i < width*height; i++) {
      const [Y, Cb, Cr] = rgbToYCbCr(data[i*channels], data[i*channels+1], data[i*channels+2]);
      Yc[i] = Y; Cbc[i] = Cb; Crc[i] = Cr;
    }

    const fY  = laplace(Yc,  width, height);
    const fCb = laplace(Cbc, width, height);
    const fCr = laplace(Crc, width, height);

    const selY  = selectLowVarianceBlocks(extractBlocks(fY,  width, height), T);
    const selCb = selectLowVarianceBlocks(extractBlocks(fCb, width, height), T);
    const selCr = selectLowVarianceBlocks(extractBlocks(fCr, width, height), T);

    const sY  = correlationStats(selY);
    const sCb = correlationStats(selCb);
    const sCr = correlationStats(selCr);

    if (!sY || !sCb || !sCr) return { error: 'not enough blocks' };

    // Combined features
    const highCorrCombined = (sY.highCorrFrac + sCb.highCorrFrac + sCr.highCorrFrac) / 3;
    const absMeanCombined  = (sY.absMean + sCb.absMean + sCr.absMean) / 3;
    const stdCombined      = (sY.std + sCb.std + sCr.std) / 3;

    return { sY, sCb, sCr, highCorrCombined, absMeanCombined, stdCombined };
  } catch (err) {
    return { error: err.message };
  }
}

const IMAGES = {
  'AI': [
    'ai/flux-grid.jpg', 'ai/flux-schnell-grid.jpg',
    'ai/sdxl-sample-01.jpg', 'ai/sdxl-test.png',
    'ai/aurora-astronaut.webp', 'ai/aurora-cyberpunk-city.webp',
    'ai/aurora-tea-dog.webp', 'ai/aurora-mountain.webp', 'ai/aurora-vangogh-cat.webp',
    'ai/aurora-cherry-blossom.webp',
    'ai/sd-txt2img-01.png', 'ai/sd-img2img-mountains.png',
    'ai/sd-golem.jpg', 'ai/sd-android.png',
    'ai/dalle-blue-man.png', 'ai/dalle-robot-letter.png',
    'ai/chatgpt-image.png', 'ai/firefly-tabby-cat.jpg',
    'ai/bing-creator-puppy.jpg', 'ai/dalle-puppy.webp',
  ],
  'Real': [
    'real/crater-lake-nocreds.jpg', 'real/ant-photo.jpg', 'real/dog-photo.jpg',
    'real/flower-macro.jpg', 'real/cloudscape.jpg', 'real/golden-gate-bridge.jpg',
    'real/cat-photo.jpg', 'real/mountain-landscape.jpg', 'real/lena.png',
    'real/nasa-astronaut-portrait.jpg', 'real/nasa-wildlife.jpg', 'real/nasa-madagascar-coast.jpg',
    'real/picsum-landscape-65.jpg', 'real/picsum-portrait-91.jpg',
    'real/waterfall.jpg', 'real/car-photo.jpg', 'real/ocean-waves.jpg', 'real/city-nyc.jpg',
  ],
};

// Collect all results
const results = [];
for (const [group, images] of Object.entries(IMAGES)) {
  for (const img of images) {
    const r = await analyzeImage(img);
    const name = img.split('/').pop();
    if (r.error) { console.error(`SKIP ${name}: ${r.error}`); continue; }
    results.push({ name, group, isAI: group === 'AI', ...r });
  }
}

// Print key features
console.log('Key features (T=30 lowest-variance blocks)\n');
console.log('image'.padEnd(35), 'highCorrComb'.padEnd(14), 'absMeanComb'.padEnd(13), 'stdComb'.padEnd(10), 'Y.skew'.padEnd(8), 'group');
console.log('-'.repeat(95));
for (const r of results) {
  console.log(
    r.name.padEnd(35),
    r.highCorrCombined.toFixed(4).padEnd(14),
    r.absMeanCombined.toFixed(4).padEnd(13),
    r.stdCombined.toFixed(4).padEnd(10),
    r.sY.skew.toFixed(3).padEnd(8),
    r.group
  );
}

// Test all features as thresholds
console.log('\n--- Threshold accuracy for each feature ---');
const features = ['highCorrCombined', 'absMeanCombined', 'stdCombined'];
for (const feat of features) {
  const vals = results.map(r => r[feat]);
  const minV = Math.min(...vals), maxV = Math.max(...vals);
  const step = (maxV - minV) / 200;

  let bestThresh = 0, bestAcc = 0, bestDir = 1;
  for (const dir of [1, -1]) { // 1: AI > threshold, -1: AI < threshold
    for (let t = minV; t <= maxV; t += step) {
      let correct = 0;
      for (const r of results) {
        const predAI = dir === 1 ? r[feat] > t : r[feat] < t;
        if (r.isAI === predAI) correct++;
      }
      const acc = correct / results.length;
      if (acc > bestAcc) { bestAcc = acc; bestThresh = t; bestDir = dir; }
    }
  }

  const aiVals = results.filter(r => r.isAI).map(r => r[feat]);
  const realVals = results.filter(r => !r.isAI).map(r => r[feat]);
  const aiMean = aiVals.reduce((s,v)=>s+v,0)/aiVals.length;
  const realMean = realVals.reduce((s,v)=>s+v,0)/realVals.length;

  console.log(`\n${feat}:`);
  console.log(`  AI mean=${aiMean.toFixed(4)}  range=[${Math.min(...aiVals).toFixed(4)}, ${Math.max(...aiVals).toFixed(4)}]`);
  console.log(`  Real mean=${realMean.toFixed(4)}  range=[${Math.min(...realVals).toFixed(4)}, ${Math.max(...realVals).toFixed(4)}]`);
  console.log(`  Best threshold: ${bestThresh.toFixed(4)} (AI ${bestDir===1?'>':'<'} thresh) → accuracy: ${(bestAcc*100).toFixed(1)}%`);
}
