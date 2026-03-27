#!/usr/bin/env node
/**
 * Diagnoses which L3 signals fire on a specific image.
 * Usage: node tests/fixtures/diagnose-image.mjs real/mountain-landscape.jpg
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = join(__dirname, 'images');

const relPath = process.argv[2] || 'real/mountain-landscape.jpg';
const fullPath = join(IMAGES_DIR, relPath);

const img = sharp(fullPath).resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true });
const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
const { width, height, channels } = info;
const totalPixels = width * height;

console.log(`Image: ${relPath}`);
console.log(`Size: ${width}x${height} (${totalPixels} pixels)`);
console.log(`Channels: ${channels}`);

const sniffedMime = relPath.endsWith('.jpg') || relPath.endsWith('.jpeg') ? 'image/jpeg' :
                    relPath.endsWith('.webp') ? 'image/webp' :
                    relPath.endsWith('.png') ? 'image/png' : 'unknown';
const isLossy = sniffedMime === 'image/jpeg' || sniffedMime === 'image/webp';
console.log(`isLossy: ${isLossy} (${sniffedMime})\n`);

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

// 1. LSB entropy
if (!isLossy) {
  const lsbCounts = [0, 0];
  for (let i = 0; i < totalPixels; i++) {
    lsbCounts[data[i * channels] & 1]++;
    lsbCounts[data[i * channels + 1] & 1]++;
    lsbCounts[data[i * channels + 2] & 1]++;
  }
  const lsbTotal = lsbCounts[0] + lsbCounts[1];
  const p0 = lsbCounts[0] / lsbTotal;
  const p1 = lsbCounts[1] / lsbTotal;
  const lsbEntropy = p0 > 0 && p1 > 0 ? -(p0 * Math.log2(p0) + p1 * Math.log2(p1)) : 0;
  console.log(`1. LSB entropy: ${lsbEntropy.toFixed(4)} (fires if < 0.85) → ${lsbEntropy < 0.85 ? 'FIRES' : 'clean'}`);
} else {
  console.log('1. LSB entropy: SKIPPED (lossy)');
}

// 2. Gradient smoothness
const sampleStep = Math.max(1, Math.floor(Math.sqrt(totalPixels / 5000)));
let gradSum = 0, gradSampleCount = 0;
for (let y = 1; y < height - 1; y += sampleStep) {
  for (let x = 1; x < width - 1; x += sampleStep) {
    const gx = -lum[(y-1)*width+(x-1)] + lum[(y-1)*width+(x+1)]
               - 2*lum[y*width+(x-1)]   + 2*lum[y*width+(x+1)]
               - lum[(y+1)*width+(x-1)] + lum[(y+1)*width+(x+1)];
    const gy = -lum[(y-1)*width+(x-1)] - 2*lum[(y-1)*width+x] - lum[(y-1)*width+(x+1)]
               + lum[(y+1)*width+(x-1)] + 2*lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)];
    gradSum += Math.sqrt(gx*gx + gy*gy);
    gradSampleCount++;
  }
}
const avgGradient = gradSampleCount > 0 ? gradSum / gradSampleCount : 0;
console.log(`2. Gradient: ${avgGradient.toFixed(4)} (fires if < 8 AND pixels > 10000) → ${avgGradient < 8 && totalPixels > 10000 ? 'FIRES' : 'clean'}`);

// 3. Color channel correlation
let sumR = 0, sumG = 0, sumB = 0;
for (let i = 0; i < totalPixels; i++) { sumR += r[i]; sumG += g[i]; sumB += b[i]; }
const meanR = sumR / totalPixels, meanG = sumG / totalPixels;
let covRG = 0, varR = 0, varG = 0;
for (let i = 0; i < totalPixels; i++) {
  covRG += (r[i] - meanR) * (g[i] - meanG);
  varR += (r[i] - meanR) ** 2;
  varG += (g[i] - meanG) ** 2;
}
const corrRG = (varR > 0 && varG > 0) ? covRG / Math.sqrt(varR * varG) : 0;
console.log(`3. R-G correlation: ${corrRG.toFixed(4)} (fires if |corr| > 0.97) → ${Math.abs(corrRG) > 0.97 ? 'FIRES' : 'clean'}`);

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
if (!isLossy) {
  console.log(`4. Noise floor: ${avgNoise.toFixed(4)} (fires if < 1.2 AND pixels > 50000) → ${avgNoise < 1.2 && totalPixels > 50000 ? 'FIRES' : 'clean'}`);
} else {
  console.log(`4. Noise floor: SKIPPED (lossy), value would be ${avgNoise.toFixed(4)}`);
}

// 5. Noise residual autocorr
const nLum = [];
for (let y = 1; y < height - 1; y += sampleStep) {
  for (let x = 1; x < width - 1; x += sampleStep) {
    const i = y * width + x;
    const neighbours = [i-width-1, i-width, i-width+1, i-1, i+1, i+width-1, i+width, i+width+1];
    const mR = neighbours.reduce((s, j) => s + r[j], 0) / 8;
    const mG = neighbours.reduce((s, j) => s + g[j], 0) / 8;
    const mB = neighbours.reduce((s, j) => s + b[j], 0) / 8;
    nLum.push(0.299*(r[i]-mR) + 0.587*(g[i]-mG) + 0.114*(b[i]-mB));
  }
}
const n = nLum.length;
let cov1 = 0, varN = 0;
const meanN = nLum.reduce((s, v) => s + v, 0) / n;
for (let i = 0; i < n - 1; i++) {
  cov1 += (nLum[i] - meanN) * (nLum[i + 1] - meanN);
  varN  += (nLum[i] - meanN) ** 2;
}
varN += (nLum[n - 1] - meanN) ** 2;
const autocorr = varN > 0 ? Math.max(0, cov1 / varN) : 0;
console.log(`5. Noise autocorr: ${autocorr.toFixed(4)} (fires if > 0.18) → ${autocorr > 0.18 ? 'FIRES' : 'clean'}`);
