#!/usr/bin/env node
/**
 * Bundle TF.js (core + WASM backend + converter) into a single ES module
 * for use in the extension service worker. Also copies WASM binary files.
 *
 * Run: node scripts/bundle-tfjs.mjs
 * Output:
 *   lib/tfjs/tfjs.min.js          (~480KB)
 *   lib/tfjs/tfjs-backend-wasm.wasm        (304KB — fallback)
 *   lib/tfjs/tfjs-backend-wasm-simd.wasm   (415KB — SIMD-accelerated)
 */

import { build } from 'esbuild';
import { copyFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const OUT_DIR = resolve(ROOT, 'lib/tfjs');

mkdirSync(OUT_DIR, { recursive: true });

// Bundle entry: CPU backend for SW (WASM requires URL.createObjectURL, unavailable in MV3 SW)
const entry = `
import '@tensorflow/tfjs-backend-cpu';
export { loadGraphModel } from '@tensorflow/tfjs-converter';
export { tensor4d, tidy, setBackend, ready } from '@tensorflow/tfjs-core';
`;

await build({
  stdin: {
    contents: entry,
    resolveDir: ROOT,
  },
  outfile: `${OUT_DIR}/tfjs.min.js`,
  bundle: true,
  minify: true,
  format: 'esm',
  platform: 'browser',
  target: ['chrome112', 'firefox115'],
  define: {
    'process.env.NODE_ENV': '"production"',
    'global': 'globalThis',
  },
  logLevel: 'info',
});

// Copy WASM binaries
const wasmSrc = resolve(ROOT, 'node_modules/@tensorflow/tfjs-backend-wasm/dist');
for (const f of [
  'tfjs-backend-wasm.wasm',
  'tfjs-backend-wasm-simd.wasm',
  'tfjs-backend-wasm-threaded-simd.wasm',
]) {
  copyFileSync(`${wasmSrc}/${f}`, `${OUT_DIR}/${f}`);
  console.log(`Copied ${f}`);
}

console.log(`\nBundle → ${OUT_DIR}/tfjs.min.js`);
