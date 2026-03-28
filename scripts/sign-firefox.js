#!/usr/bin/env node
/**
 * Signs the extension for Firefox self-distribution via Mozilla AMO.
 *
 * Reads credentials from .env:
 *   MOZILLA_JWT_ISSUER  — your AMO API key (user:xxxxx)
 *   MOZILLA_JWT_SECRET  — your AMO API secret
 *
 * Produces: dist/lens-{version}.xpi (Mozilla-signed, self-hosted channel)
 *
 * Users can install the .xpi directly from a download link — no Firefox
 * store listing or review queue required.
 */

import { execSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// Load .env
const envPath = join(ROOT, '.env');
if (!existsSync(envPath)) {
  console.error('Error: .env file not found.');
  console.error('Create .env with MOZILLA_JWT_ISSUER and MOZILLA_JWT_SECRET.');
  console.error('Get credentials at: https://addons.mozilla.org/developers/addon/api/key/');
  process.exit(1);
}

const env = Object.fromEntries(
  readFileSync(envPath, 'utf8')
    .split('\n')
    .filter(l => l.includes('=') && !l.startsWith('#'))
    .map(l => l.split('=').map(s => s.trim()))
);

const apiKey = env.MOZILLA_JWT_ISSUER;
const apiSecret = env.MOZILLA_JWT_SECRET;

if (!apiKey || !apiSecret) {
  console.error('Error: MOZILLA_JWT_ISSUER and MOZILLA_JWT_SECRET must be set in .env');
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(ROOT, 'manifest.json'), 'utf8'));
const version = manifest.version;

console.log(`Signing Lens v${version} for Firefox (unlisted/self-hosted)...`);

const IGNORE = [
  'tests', 'scripts', 'node_modules', 'dist', 'chrome',
  '.browser-profile', '.lens-profile', 'CLAUDE.md',
  'vitest.config.js', 'agent-browser.json', '.env', '.git',
  'package-lock.json', 'lens-ext',
].map(f => `--ignore-files="${f}"`).join(' ');

const cmd = [
  'npx web-ext sign',
  '--channel=unlisted',
  `--source-dir="${ROOT}"`,
  `--artifacts-dir="${join(ROOT, 'dist')}"`,
  `--api-key="${apiKey}"`,
  `--api-secret="${apiSecret}"`,
  IGNORE,
].join(' ');

execSync(cmd, { stdio: 'inherit', cwd: ROOT });

console.log(`\nSigned XPI saved to dist/`);
console.log(`Host it at e.g. https://thevgergroup.com/lens/lens-${version}.xpi`);
console.log(`Link to it from https://thevgergroup.com/blog/lens`);
