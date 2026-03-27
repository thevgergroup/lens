#!/usr/bin/env node
/**
 * Downloads fixture images for the LENS test suite.
 *
 * Run once before running tests:
 *   node tests/fixtures/download-fixtures.js
 *
 * What this downloads:
 *   - C2PA sample images from contentauth (known AI + real photos with provenance)
 *   - Generates minimal synthetic images for edge-case testing
 *
 * For DiffusionDB samples (optional, requires Python + huggingface_hub):
 *   python3 tests/fixtures/download-diffusiondb.py
 */

import https from 'https';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMAGES_DIR = path.join(__dirname, 'images');

const C2PA_BASE = 'https://contentauth.github.io/example-assets/images';

const FIXTURES = [
  // AI-generated images with C2PA provenance (high confidence ground truth)
  {
    url: `${C2PA_BASE}/ChatGPT_Image.png`,
    dest: 'ai/chatgpt-image.png',
    description: 'ChatGPT generated image with C2PA content credentials',
  },
  {
    url: `${C2PA_BASE}/Firefly_tabby_cat.jpg`,
    dest: 'ai/firefly-tabby-cat.jpg',
    description: 'Adobe Firefly generated cat with C2PA content credentials',
  },

  // Real photographs with C2PA provenance (true negatives)
  {
    url: `${C2PA_BASE}/crater-lake-cr.jpg`,
    dest: 'real/crater-lake.jpg',
    description: 'Real photograph edited in Lightroom with C2PA credentials',
  },
  {
    url: `${C2PA_BASE}/car-es-Ps-Cr.jpg`,
    dest: 'real/car-photo.jpg',
    description: 'Real car photograph edited in Photoshop with C2PA credentials',
  },
  {
    url: `${C2PA_BASE}/cloudscape-ACA-Cr.jpeg`,
    dest: 'real/cloudscape.jpg',
    description: 'Real cloudscape with C2PA credentials from Adobe Content Authenticity',
  },
];

function download(url, destPath) {
  return new Promise((resolve, reject) => {
    const fullPath = path.join(IMAGES_DIR, destPath);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });

    if (fs.existsSync(fullPath)) {
      console.log(`  ✓ Already exists: ${destPath}`);
      resolve();
      return;
    }

    const client = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(fullPath);

    const request = client.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        file.close();
        fs.unlinkSync(fullPath);
        download(response.headers.location, destPath).then(resolve).catch(reject);
        return;
      }
      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(fullPath);
        reject(new Error(`HTTP ${response.statusCode} for ${url}`));
        return;
      }
      response.pipe(file);
      file.on('finish', () => file.close(resolve));
    });

    request.on('error', (err) => {
      fs.unlink(fullPath, () => {});
      reject(err);
    });
  });
}

/**
 * Creates minimal synthetic images for edge case tests.
 * These are valid image files with known properties — no external download needed.
 */
function createSyntheticFixtures() {
  // 1x1 PNG tracking pixel (should be ignored by extension — too small)
  const trackingPixelPath = path.join(IMAGES_DIR, 'edge/tracking-pixel.png');
  fs.mkdirSync(path.dirname(trackingPixelPath), { recursive: true });
  if (!fs.existsSync(trackingPixelPath)) {
    // Minimal valid 1x1 red PNG (hand-crafted bytes)
    const png1x1 = Buffer.from(
      '89504e470d0a1a0a0000000d49484452000000010000000108020000009001' +
      '2e00000000c4944415478016360f8cfc00000000200016e12150000000049454e44ae426082',
      'hex'
    );
    fs.writeFileSync(trackingPixelPath, png1x1);
    console.log('  ✓ Created: edge/tracking-pixel.png (1x1, should be ignored)');
  }

  // 24x24 PNG icon (should be ignored by extension — under 32px threshold)
  const iconPath = path.join(IMAGES_DIR, 'edge/icon-small.png');
  if (!fs.existsSync(iconPath)) {
    // Copy tracking pixel and rename — same effect for size filtering test
    fs.copyFileSync(trackingPixelPath, iconPath);
    console.log('  ✓ Created: edge/icon-small.png (tiny icon, should be ignored)');
  }

  // Create symlinks/copies for URL-heuristic tests that need AI-named filenames
  const urlHeuristicFiles = [
    { src: 'ai/diffusiondb-000.png', dest: 'ai/DALL-E-sample.png' },
    { src: 'ai/diffusiondb-000.png', dest: 'ai/MJ-abc123def456.png' },
  ];

  for (const { src, dest } of urlHeuristicFiles) {
    const srcPath = path.join(IMAGES_DIR, src);
    const destPath = path.join(IMAGES_DIR, dest);
    if (fs.existsSync(srcPath) && !fs.existsSync(destPath)) {
      fs.copyFileSync(srcPath, destPath);
      console.log(`  ✓ Created: ${dest} (copy for URL heuristic test)`);
    }
  }
}

async function main() {
  console.log('Downloading LENS test fixtures...\n');

  console.log('C2PA sample images (contentauth.github.io):');
  for (const fixture of FIXTURES) {
    process.stdout.write(`  Downloading ${fixture.dest}... `);
    try {
      await download(fixture.url, fixture.dest);
      console.log('✓');
    } catch (err) {
      console.log(`✗ FAILED: ${err.message}`);
      console.log(`    ${fixture.description}`);
      console.log(`    Manual download: ${fixture.url}`);
    }
  }

  console.log('\nCreating synthetic edge-case fixtures:');
  createSyntheticFixtures();

  console.log('\nFixture setup complete.');
  console.log('\nFor Stable Diffusion pixel-stats test images, run:');
  console.log('  python3 tests/fixtures/download-diffusiondb.py');
  console.log('\nFor SynthID/FFT test images:');
  console.log('  Generate images at https://aitestkitchen.withgoogle.com (ImageFX)');
  console.log('  Save to: tests/fixtures/images/ai/synthid-*.png');
}

main();
