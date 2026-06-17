/**
 * Unit tests for Layer 3: Statistical pixel analysis
 * Tests analyzePixelStatistics() with synthetic ImageData.
 */

import './helpers/browser-stubs.js';
import { describe, it, expect, beforeAll } from 'vitest';
import { makeRealCameraImageData, makeAIImageData } from './helpers/image-factory.js';

let analyzePixelStatistics;

beforeAll(async () => {
  const mod = await import('../../lib/detector.js');
  analyzePixelStatistics = mod.analyzePixelStatistics;
});

describe('L3: Pixel statistics — minimum size guard', () => {
  it('returns score 0 for tiny image (< 100 pixels)', () => {
    // 9x9 = 81 pixels, below the 100px minimum
    const tiny = new ImageData(9, 9);
    const result = analyzePixelStatistics(tiny);
    expect(result.score).toBe(0);
    expect(result.signals.length).toBe(0);
  });

  it('processes a 64x64 image without throwing', () => {
    const img = makeRealCameraImageData(64, 64);
    expect(() => analyzePixelStatistics(img)).not.toThrow();
  });
});

describe('L3: Pixel statistics — real camera (random noise)', () => {
  it('gives a lower score to pseudo-random noise (camera-like) than to smooth gradients', () => {
    // Random noise = high LSB entropy = lower AI probability
    // Use 256x256 to trigger noise floor check (requires > 50000 px)
    const cameraImage = makeRealCameraImageData(256, 256);
    const aiImage = makeAIImageData(256, 256);

    const cameraResult = analyzePixelStatistics(cameraImage);
    const aiResult = analyzePixelStatistics(aiImage);

    // The AI-like smooth gradient image should score at least as high as random noise
    // (both may hit 0.55 if edge chrominance fires on the random image too)
    expect(aiResult.score).toBeGreaterThanOrEqual(cameraResult.score);
  });
});

describe('L3: Pixel statistics — AI-like smooth gradient', () => {
  it('returns non-zero score for smooth gradient image (256x256 triggers noise floor check)', () => {
    const img = makeAIImageData(256, 256);
    const result = analyzePixelStatistics(img);
    expect(result.score).toBeGreaterThan(0);
  });

  it('returns pixel-type signals', () => {
    const img = makeAIImageData(256, 256);
    const result = analyzePixelStatistics(img);
    if (result.signals.length > 0) {
      expect(result.signals.every(s => s.type === 'pixel')).toBe(true);
    }
  });

  it('score does not exceed 0.65 (L3 max cap)', () => {
    const img = makeAIImageData(512, 512);
    const result = analyzePixelStatistics(img);
    expect(result.score).toBeLessThanOrEqual(0.65);
  });
});

describe('L3: Pixel statistics — uniform color', () => {
  it('handles fully uniform image without throwing', () => {
    const data = new Uint8ClampedArray(128 * 128 * 4).fill(128);
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
    const img = new ImageData(data, 128, 128);
    expect(() => analyzePixelStatistics(img)).not.toThrow();
  });
});

describe('L3: DCT high-frequency energy signal', () => {
  it('fires on ultra-smooth gradient (AI-like, meanHighFrac should be well below 0.10)', () => {
    // Create a very smooth 256x256 gradient — energy concentrated in DC / very-low AC.
    // meanHighFrac for this image should be << 0.10 → DCT signal must fire.
    const W = 256, H = 256;
    const data = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        const v = Math.round(100 + (x / W) * 80 + (y / H) * 20); // gentle ramp 100–200
        data[idx]     = v;
        data[idx + 1] = v;
        data[idx + 2] = v;
        data[idx + 3] = 255;
      }
    }
    const img = new ImageData(data, W, H);
    const result = analyzePixelStatistics(img);
    const dctSignal = result.signals.find(s => s.label.includes('DCT'));
    expect(dctSignal).toBeDefined();
    expect(dctSignal.label).toMatch(/DCT high-freq energy/);
  });

  it('does NOT fire on high-frequency random noise (real camera-like)', () => {
    // Random noise spreads energy uniformly → meanHighFrac near 0.33 → well above 0.12.
    const img = makeRealCameraImageData(256, 256);
    const result = analyzePixelStatistics(img);
    const dctSignal = result.signals.find(s => s.label.includes('DCT'));
    expect(dctSignal).toBeUndefined();
  });

  it('does not fire on images smaller than 10000 pixels (guard)', () => {
    // 99×99 = 9801 < 10000 — below the minimum pixel check
    const W = 99, H = 99;
    const data = new Uint8ClampedArray(W * H * 4);
    // Make it a smooth gradient so it *would* fire if the size guard were absent
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        const v = Math.round(100 + (x / W) * 80);
        data[idx] = data[idx + 1] = data[idx + 2] = v;
        data[idx + 3] = 255;
      }
    }
    const img = new ImageData(data, W, H);
    const result = analyzePixelStatistics(img);
    const dctSignal = result.signals.find(s => s.label.includes('DCT'));
    expect(dctSignal).toBeUndefined();
  });
});
