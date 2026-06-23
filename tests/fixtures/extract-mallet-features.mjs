#!/usr/bin/env node
/**
 * Extract Mallet et al. (2025) noise correlation features from fixture images.
 *
 * For each image: applies Laplace filter to YCbCr channels, selects T blocks
 * with jointly lowest intra-mean AND intra-variance (paper eq. 2), computes
 * the full lower-triangle Pearson correlation matrix, and outputs the feature
 * vector as a row in features.jsonl.
 *
 * Output: tests/fixtures/mallet-features.jsonl
 *   Each line: { "path": "...", "label": 0|1, "features": [...1305 floats...] }
 *
 * Usage:
 *   node tests/fixtures/extract-mallet-features.mjs
 *   node tests/fixtures/extract-mallet-features.mjs --max-per-class 300
 */

import sharp from 'sharp';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync, createWriteStream, existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES   = join(__dirname, 'images');
const OUTPUT     = join(__dirname, 'mallet-features.jsonl');

const T = 20; // blocks per channel — paper uses 30 but we have fewer training images

const args  = process.argv.slice(2);
const maxArg = args.indexOf('--max-per-class');
const MAX_PER_CLASS = maxArg >= 0 ? parseInt(args[maxArg + 1]) : Infinity;

// ── image collection ─────────────────────────────────────────────────────────

function collectImages(dir, label) {
  const results = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st   = statSync(full);
    if (st.isDirectory()) {
      results.push(...collectImages(full, label));
    } else if (/\.(jpg|jpeg|png|webp)$/i.test(entry)) {
      results.push({ path: full, label });
    }
  }
  return results;
}

const aiImages   = collectImages(join(FIXTURES, 'ai'),   1).slice(0, MAX_PER_CLASS);
const realImages = collectImages(join(FIXTURES, 'real'), 0).slice(0, MAX_PER_CLASS);
const all        = [...aiImages, ...realImages];

console.log(`AI: ${aiImages.length}  Real: ${realImages.length}  Total: ${all.length}`);
console.log(`T=${T} → feature vector = ${3 * T*(T-1)/2} per image`);
console.log(`Output: ${OUTPUT}\n`);

// ── feature extraction ───────────────────────────────────────────────────────

function laplace(ch, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = -4*ch[i] + ch[i-w] + ch[i+w] + ch[i-1] + ch[i+1];
    }
  }
  return out;
}

function extractBlocks(filtered, w, h) {
  const blks = [];
  for (let by = 1; by + 8 < h - 1; by += 8) {
    for (let bx = 1; bx + 8 < w - 1; bx += 8) {
      const blk = new Float32Array(64);
      for (let dy = 0; dy < 8; dy++)
        for (let dx = 0; dx < 8; dx++)
          blk[dy*8+dx] = filtered[(by+dy)*w + (bx+dx)];
      blks.push(blk);
    }
  }
  return blks;
}

function blockStats(blk) {
  let s = 0;
  for (const v of blk) s += v;
  const mean = s / 64;
  let v = 0;
  for (const x of blk) v += (x - mean) * (x - mean);
  return { mean: Math.abs(mean), variance: v / 64 };
}

// Paper eq. 2: select T blocks with jointly lowest intra-mean AND intra-variance.
// Union of T-lowest-variance and T-lowest-mean, then take overall top-T by combined rank.
function selectBlocks(blks) {
  const stats = blks.map((b, i) => ({ i, ...blockStats(b) }))
    .filter(s => s.variance > 0.01); // skip flat/zero blocks

  if (stats.length < T) return null;

  // Rank by variance ascending, rank by |mean| ascending, sum ranks
  const byVar  = [...stats].sort((a, b) => a.variance - b.variance);
  const byMean = [...stats].sort((a, b) => a.mean     - b.mean);

  const varRank  = new Map(byVar.map( (s, r) => [s.i, r]));
  const meanRank = new Map(byMean.map((s, r) => [s.i, r]));

  return stats
    .map(s => ({ i: s.i, rank: varRank.get(s.i) + meanRank.get(s.i) }))
    .sort((a, b) => a.rank - b.rank)
    .slice(0, T)
    .map(s => blks[s.i]);
}

function pearson(a, b) {
  let sa = 0, sb = 0;
  for (let i = 0; i < 64; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa/64, mb = sb/64;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < 64; i++) {
    const da = a[i]-ma, db = b[i]-mb;
    cov += da*db; va += da*da; vb += db*db;
  }
  return (va > 1e-10 && vb > 1e-10) ? cov / Math.sqrt(va*vb) : 0;
}

// Returns lower-triangle of T×T correlation matrix as flat array (T*(T-1)/2 values)
function correlationTriangle(sel) {
  const tri = [];
  for (let i = 1; i < sel.length; i++)
    for (let j = 0; j < i; j++)
      tri.push(pearson(sel[i], sel[j]));
  return tri;
}

async function extractFeatures(imgPath) {
  const { data, info } = await sharp(imgPath)
    .resize({ width: 512, height: 512, fit: 'inside', withoutEnlargement: true })
    .toFormat('raw')
    .toBuffer({ resolveWithObject: true });

  const { width: w, height: h, channels } = info;
  const n = w * h;

  const Y  = new Float32Array(n);
  const Cb = new Float32Array(n);
  const Cr = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const r = data[i * channels];
    const g = data[i * channels + 1];
    const b = data[i * channels + 2];
    Y[i]  =  0.299   * r + 0.587   * g + 0.114   * b;
    Cb[i] = -0.16874 * r - 0.33126 * g + 0.5     * b + 128;
    Cr[i] =  0.5     * r - 0.41869 * g - 0.08131 * b + 128;
  }

  const selY  = selectBlocks(extractBlocks(laplace(Y,  w, h), w, h));
  const selCb = selectBlocks(extractBlocks(laplace(Cb, w, h), w, h));
  const selCr = selectBlocks(extractBlocks(laplace(Cr, w, h), w, h));

  if (!selY || !selCb || !selCr) return null;

  return [
    ...correlationTriangle(selY),
    ...correlationTriangle(selCb),
    ...correlationTriangle(selCr),
  ];
}

// ── main ─────────────────────────────────────────────────────────────────────

const out = createWriteStream(OUTPUT);
let ok = 0, skip = 0;

for (const { path, label } of all) {
  try {
    const features = await extractFeatures(path);
    if (!features) { skip++; continue; }
    out.write(JSON.stringify({ path, label, features }) + '\n');
    ok++;
    if (ok % 50 === 0) process.stdout.write(`  ${ok}/${all.length} extracted\r`);
  } catch (e) {
    skip++;
    if (process.env.VERBOSE) console.warn(`  skip ${path}: ${e.message}`);
  }
}

out.end();
console.log(`\nDone: ${ok} extracted, ${skip} skipped → ${OUTPUT}`);
