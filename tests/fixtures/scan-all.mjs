#!/usr/bin/env node
/**
 * Quick scan of all fixture images against L3 signals.
 * Usage: node tests/fixtures/scan-all.mjs
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

function sniffIsLossy(relPath) {
  const p = relPath.toLowerCase();
  return p.endsWith('.jpg') || p.endsWith('.jpeg') || p.endsWith('.webp');
}

async function analyzeImage(relPath) {
  const fullPath = join(IMAGES_DIR, relPath);
  try {
    const img = sharp(fullPath).resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true });
    const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    const totalPixels = width * height;
    const isLossy = sniffIsLossy(relPath);

    const r = new Float32Array(totalPixels);
    const g = new Float32Array(totalPixels);
    const b = new Float32Array(totalPixels);
    const lum = new Float32Array(totalPixels);

    for (let i = 0; i < totalPixels; i++) {
      const base = i * channels;
      r[i] = data[base];
      g[i] = data[base + 1];
      b[i] = data[base + 2];
      lum[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
    }

    const signals = [];
    let score = 0;
    const sampleStep = Math.max(1, Math.floor(Math.sqrt(totalPixels / 5000)));

    // 1. LSB entropy
    if (!isLossy) {
      const lsbCounts = [0, 0];
      for (let i = 0; i < totalPixels; i++) {
        lsbCounts[data[i * channels] & 1]++;
        lsbCounts[data[i * channels + 1] & 1]++;
        lsbCounts[data[i * channels + 2] & 1]++;
      }
      const lsbTotal = lsbCounts[0] + lsbCounts[1];
      const p0 = lsbCounts[0] / lsbTotal, p1 = lsbCounts[1] / lsbTotal;
      const lsbEntropy = p0 > 0 && p1 > 0 ? -(p0 * Math.log2(p0) + p1 * Math.log2(p1)) : 0;
      if (lsbEntropy < 0.85) { signals.push(`LSB(${lsbEntropy.toFixed(2)})`); score = Math.max(score, 0.55); }
    }

    // 2. Gradient
    let gradSum = 0, gradCount = 0;
    for (let y = 1; y < height - 1; y += sampleStep) {
      for (let x = 1; x < width - 1; x += sampleStep) {
        const gx = -lum[(y-1)*width+(x-1)] + lum[(y-1)*width+(x+1)] - 2*lum[y*width+(x-1)] + 2*lum[y*width+(x+1)] - lum[(y+1)*width+(x-1)] + lum[(y+1)*width+(x+1)];
        const gy = -lum[(y-1)*width+(x-1)] - 2*lum[(y-1)*width+x] - lum[(y-1)*width+(x+1)] + lum[(y+1)*width+(x-1)] + 2*lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)];
        gradSum += Math.sqrt(gx*gx + gy*gy);
        gradCount++;
      }
    }
    const avgGrad = gradCount > 0 ? gradSum / gradCount : 0;
    if (avgGrad < 8 && totalPixels > 10000) { signals.push(`grad(${avgGrad.toFixed(1)})`); score = Math.max(score, 0.45); }

    // 3. R-G correlation
    let sumR = 0, sumG = 0;
    for (let i = 0; i < totalPixels; i++) { sumR += r[i]; sumG += g[i]; }
    const mR = sumR / totalPixels, mG = sumG / totalPixels;
    let covRG = 0, vR = 0, vG = 0;
    for (let i = 0; i < totalPixels; i++) {
      covRG += (r[i] - mR) * (g[i] - mG);
      vR += (r[i] - mR) ** 2;
      vG += (g[i] - mG) ** 2;
    }
    const corrRG = (vR > 0 && vG > 0) ? covRG / Math.sqrt(vR * vG) : 0;
    if (Math.abs(corrRG) > 0.97) { signals.push(`RG(${corrRG.toFixed(3)})`); score = Math.max(score, 0.4); }

    // 4. Noise floor
    let noiseSum = 0, noiseSamples = 0;
    for (let y = 1; y < height - 1; y += sampleStep * 2) {
      for (let x = 1; x < width - 1; x += sampleStep * 2) {
        const center = lum[y * width + x];
        const avg = (lum[(y-1)*width+x] + lum[(y+1)*width+x] + lum[y*width+(x-1)] + lum[y*width+(x+1)]) / 4;
        noiseSum += Math.abs(center - avg);
        noiseSamples++;
      }
    }
    const avgNoise = noiseSamples > 0 ? noiseSum / noiseSamples : 0;
    if (!isLossy && avgNoise < 1.2 && totalPixels > 50000) { signals.push(`noise(${avgNoise.toFixed(3)})`); score = Math.max(score, 0.5); }

    // 5. Noise residual autocorr — disabled: overlaps with real photos (cloudscape JPEG = 0.255)

    const level = score >= 0.45 ? 'POSSIBLE' : score >= 0.20 ? 'UNLIKELY' : 'CLEAN';
    return { level, score, signals };
  } catch (err) {
    return { level: 'ERROR', score: 0, signals: [err.message] };
  }
}

const AI_IMAGES = readdirSync(join(IMAGES_DIR, 'ai')).filter(f => /\.(jpg|jpeg|png|webp)$/.test(f)).map(f => `ai/${f}`);
const REAL_IMAGES = readdirSync(join(IMAGES_DIR, 'real')).filter(f => /\.(jpg|jpeg|png|webp)$/.test(f)).map(f => `real/${f}`);

let aiFalseNeg = 0, realFalsePos = 0;

console.log('=== AI IMAGES (should be POSSIBLE or higher) ===');
for (const img of AI_IMAGES.sort()) {
  const r = await analyzeImage(img);
  const name = img.split('/').pop();
  const ok = r.level !== 'CLEAN';
  if (!ok) aiFalseNeg++;
  console.log(`  [${ok ? 'OK' : 'MISS'}] ${name.padEnd(35)} ${r.level.padEnd(10)} score:${r.score.toFixed(2)}  ${r.signals.join(', ')}`);
}

console.log('\n=== REAL IMAGES (should be CLEAN or UNLIKELY) ===');
for (const img of REAL_IMAGES.sort()) {
  const r = await analyzeImage(img);
  const name = img.split('/').pop();
  const ok = r.level === 'CLEAN' || r.level === 'UNLIKELY';
  if (!ok) realFalsePos++;
  console.log(`  [${ok ? 'OK' : 'FP!'}] ${name.padEnd(35)} ${r.level.padEnd(10)} score:${r.score.toFixed(2)}  ${r.signals.join(', ')}`);
}

console.log(`\nSummary: ${aiFalseNeg} AI missed (score < 0.45), ${realFalsePos} real false positives (score ≥ 0.45)`);
