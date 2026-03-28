#!/usr/bin/env node
/**
 * Packages the extension for store submission.
 *
 * Produces: dist/lens-{version}.zip
 *
 * Includes only the files the browser needs to run the extension:
 *   manifest.json, background/, content/, lib/, popup/, icons/
 *
 * Excludes all dev tooling: tests/, node_modules/, scripts/, CLAUDE.md, etc.
 */

import { execSync } from 'child_process';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const manifest = JSON.parse(readFileSync(join(ROOT, 'manifest.json'), 'utf8'));
const version = manifest.version;
const outDir = join(ROOT, 'dist');
const zipName = `lens-${version}.zip`;
const zipPath = join(outDir, zipName);

// Files/dirs to include in the package
const INCLUDE = [
  'manifest.json',
  'background',
  'content',
  'lib',
  'popup',
  'icons',
];

// Clean and recreate dist/
if (existsSync(outDir)) rmSync(outDir, { recursive: true });
mkdirSync(outDir);

// Build the zip using the system zip command (available on macOS and Linux)
const includeArgs = INCLUDE.map(f => `"${f}"`).join(' ');
const cmd = `cd "${ROOT}" && zip -r "${zipPath}" ${includeArgs} -x "*.DS_Store" -x "__MACOSX/*"`;

console.log(`Packaging Lens v${version}...`);
execSync(cmd, { stdio: 'inherit' });

// Print size
const bytes = parseInt(execSync(`wc -c < "${zipPath}"`).toString().trim());
const kb = (bytes / 1024).toFixed(1);
console.log(`\nCreated: dist/${zipName} (${kb} KB)`);
console.log('\nReady for:');
console.log('  Chrome Web Store  → https://chrome.google.com/webstore/devconsole');
console.log('  Edge Add-ons      → https://partner.microsoft.com/dashboard/microsoftedge');
console.log('  Firefox AMO       → npm run sign:firefox');
