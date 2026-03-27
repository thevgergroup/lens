/**
 * Unit tests for Layer 2: EXIF/XMP/IPTC metadata parsing
 * Tests parseMetadata() with synthetic and real fixture images.
 */

import './helpers/browser-stubs.js';
import { describe, it, expect, beforeAll } from 'vitest';
import {
  makeMinimalJpegBuffer,
  makeJpegWithExifSoftware,
  makePngWithAIParameters,
} from './helpers/image-factory.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_AI = path.join(__dirname, '../fixtures/images/ai');
const FIXTURES_REAL = path.join(__dirname, '../fixtures/images/real');

let parseMetadata;

beforeAll(async () => {
  const mod = await import('../../lib/detector.js');
  parseMetadata = mod.parseMetadata;
});

describe('L2: Metadata — clean JPEG (no EXIF)', () => {
  it('returns score 0 for a minimal JPEG with no metadata', async () => {
    const result = await parseMetadata(makeMinimalJpegBuffer());
    expect(result.score).toBe(0);
    expect(result.signals.length).toBe(0);
  });
});

describe('L2: Metadata — JPEG EXIF Software tag', () => {
  const aiSoftwareNames = [
    'Adobe Firefly',
    'Midjourney',
    'DALL-E',
    'Stable Diffusion',
    'ComfyUI',
    'AUTOMATIC1111',
    'NovelAI',
    'Imagen',
  ];

  for (const software of aiSoftwareNames) {
    it(`flags Software="${software}"`, async () => {
      const buf = makeJpegWithExifSoftware(software);
      const result = await parseMetadata(buf);
      expect(result.score).toBeGreaterThan(0.5);
      expect(result.signals.some(s => s.type === 'exif')).toBe(true);
    });
  }

  it('does not flag Software="Adobe Photoshop 2026"', async () => {
    const buf = makeJpegWithExifSoftware('Adobe Photoshop 2026');
    const result = await parseMetadata(buf);
    // Photoshop itself is not an AI generator — score should be low or zero
    expect(result.score).toBeLessThan(0.5);
  });
});

describe('L2: Metadata — PNG tEXt AI parameters', () => {
  it('flags PNG with SD-style parameters chunk', async () => {
    const buf = makePngWithAIParameters('a beautiful sunset over mountains');
    const result = await parseMetadata(buf);
    expect(result.score).toBeGreaterThan(0.5);
    expect(result.signals.some(s => s.type === 'png-meta')).toBe(true);
  });
});

describe('L2: Metadata — real fixture files (C2PA)', () => {
  it('flags ChatGPT image with high confidence (C2PA signed)', async () => {
    const fixturePath = path.join(FIXTURES_AI, 'chatgpt-image.png');
    if (!fs.existsSync(fixturePath)) {
      console.warn('    ⚠ Fixture not found, skipping: run npm run fixture:download');
      return;
    }
    const buf = fs.readFileSync(fixturePath).buffer;
    const result = await parseMetadata(buf);
    expect(result.score).toBeGreaterThan(0.5);
    expect(result.signals.length).toBeGreaterThan(0);
  });

  it('flags Adobe Firefly image with high confidence (C2PA signed)', async () => {
    const fixturePath = path.join(FIXTURES_AI, 'firefly-tabby-cat.jpg');
    if (!fs.existsSync(fixturePath)) {
      console.warn('    ⚠ Fixture not found, skipping: run npm run fixture:download');
      return;
    }
    const buf = fs.readFileSync(fixturePath).buffer;
    const result = await parseMetadata(buf);
    expect(result.score).toBeGreaterThan(0.5);
  });

  it('does not flag Crater Lake real photo as AI', async () => {
    const fixturePath = path.join(FIXTURES_REAL, 'crater-lake.jpg');
    if (!fs.existsSync(fixturePath)) {
      console.warn('    ⚠ Fixture not found, skipping: run npm run fixture:download');
      return;
    }
    const buf = fs.readFileSync(fixturePath).buffer;
    const result = await parseMetadata(buf);
    // Real photo — may have C2PA but should NOT have AI signals
    expect(result.score).toBeLessThan(0.5);
  });
});
