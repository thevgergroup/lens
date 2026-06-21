#!/usr/bin/env node
/**
 * Scan fixture images using the actual detector.js analyzePixelStatistics()
 * plus a simple metadata check, to measure real precision/recall.
 */
import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync, readFileSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(__dirname, 'images');

// --- stub browser globals so detector.js loads in Node ---
global.ImageData = class ImageData {
  constructor(data, width, height) {
    this.data = data; this.width = width; this.height = height;
  }
};

const { analyzePixelStatistics, analyzeMetadata } = await import('../../lib/detector.js');

function collectImages(dir, prefix) {
  const results = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const rel  = `${prefix}/${entry}`;
    if (statSync(full).isDirectory()) results.push(...collectImages(full, rel));
    else if (/\.(jpg|jpeg|png|webp)$/.test(entry)) results.push({ rel, full });
  }
  return results;
}

async function analyze(fullPath) {
  const buf = readFileSync(fullPath);
  
  // L2: metadata
  let metaResult = { score: 0, signals: [] };
  try { metaResult = analyzeMetadata(buf.buffer); } catch {}

  // L3: pixel stats via sharp decode
  // sharp returns RGB (3 channels); analyzePixelStatistics expects RGBA (stride 4).
  // Convert to RGBA by inserting alpha=255 so channel indexing is correct.
  let pixelResult = { score: 0, signals: [] };
  try {
    const { data, info } = await sharp(fullPath)
      .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
      .toFormat('raw')
      .toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4]     = data[i * channels];
      rgba[i * 4 + 1] = data[i * channels + 1];
      rgba[i * 4 + 2] = data[i * channels + 2];
      rgba[i * 4 + 3] = 255;
    }
    const imgData = new ImageData(rgba, width, height);
    pixelResult = analyzePixelStatistics(imgData);
  } catch {}

  const score = Math.max(metaResult.score, pixelResult.score);
  const signals = [...metaResult.signals, ...pixelResult.signals];
  return { score, signals, metaScore: metaResult.score, pixelScore: pixelResult.score };
}

const AI_IMAGES   = collectImages(join(FIXTURES, 'ai'),   'ai');
const REAL_IMAGES = collectImages(join(FIXTURES, 'real'), 'real');

// Score levels: >=0.45 = flagged, <0.45 = not flagged
const THRESHOLD = 0.45;

console.log('\n=== AI IMAGES (want score >= 0.45) ===');
console.log('image'.padEnd(42) + ' L2'.padEnd(7) + ' L3'.padEnd(7) + ' final  signals');
console.log('-'.repeat(100));

const aiRows = [];
for (const { rel, full } of AI_IMAGES.sort((a,b) => a.rel.localeCompare(b.rel))) {
  const r = await analyze(full);
  const name = rel.replace('ai/', '');
  const flagged = r.score >= THRESHOLD;
  aiRows.push({ name, ...r, flagged });
  const sigStr = r.signals.map(s => s.label.split(':')[0]).join(', ');
  console.log(
    (flagged ? '  [OK] ' : ' [MISS] ') + name.padEnd(35),
    r.metaScore.toFixed(2).padEnd(6),
    r.pixelScore.toFixed(2).padEnd(6),
    r.score.toFixed(2).padEnd(7),
    sigStr
  );
}

console.log('\n=== REAL IMAGES (want score < 0.45) ===');
console.log('image'.padEnd(42) + ' L2'.padEnd(7) + ' L3'.padEnd(7) + ' final  signals');
console.log('-'.repeat(100));

const realRows = [];
for (const { rel, full } of REAL_IMAGES.sort((a,b) => a.rel.localeCompare(b.rel))) {
  const r = await analyze(full);
  const name = rel.replace('real/', '');
  const fp = r.score >= THRESHOLD;
  realRows.push({ name, ...r, fp });
  const sigStr = r.signals.map(s => s.label.split(':')[0]).join(', ');
  console.log(
    (fp ? '  [FP!] ' : '  [OK]  ') + name.padEnd(35),
    r.metaScore.toFixed(2).padEnd(6),
    r.pixelScore.toFixed(2).padEnd(6),
    r.score.toFixed(2).padEnd(7),
    sigStr
  );
}

// Summary
const tp = aiRows.filter(r => r.flagged).length;
const fn = aiRows.filter(r => !r.flagged).length;
const fp = realRows.filter(r => r.fp).length;
const tn = realRows.filter(r => !r.fp).length;
const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
const recall    = tp / aiRows.length;
const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

console.log('\n=== SUMMARY (threshold >= 0.45 = flagged) ===');
console.log(`AI images:   ${tp} detected / ${aiRows.length} total  (${fn} missed)`);
console.log(`Real images: ${fp} false positives / ${realRows.length} total`);
console.log(`Precision: ${(precision*100).toFixed(0)}%  Recall: ${(recall*100).toFixed(0)}%  F1: ${(f1*100).toFixed(0)}%`);

// Break down by subset
const grokRows  = aiRows.filter(r => r.name.startsWith('grok/'));
const otherAI   = aiRows.filter(r => !r.name.startsWith('grok/'));
const twitterReal = realRows.filter(r => r.name.startsWith('twitter/'));
const otherReal   = realRows.filter(r => !r.name.startsWith('twitter/'));

console.log('\n--- By subset ---');
console.log(`Grok CDN images:    ${grokRows.filter(r=>r.flagged).length}/${grokRows.length} detected`);
console.log(`Other AI images:    ${otherAI.filter(r=>r.flagged).length}/${otherAI.length} detected`);
console.log(`Twitter real photos: ${twitterReal.filter(r=>r.fp).length}/${twitterReal.length} false positives`);
console.log(`Other real photos:  ${otherReal.filter(r=>r.fp).length}/${otherReal.length} false positives`);
