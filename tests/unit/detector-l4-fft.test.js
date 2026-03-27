/**
 * Unit tests for Layer 4: FFT frequency analysis (SynthID detection)
 * Tests analyzeFrequencyDomain() with synthetic ImageData.
 */

import './helpers/browser-stubs.js';
import { describe, it, expect, beforeAll } from 'vitest';
import { makeRealCameraImageData } from './helpers/image-factory.js';

let analyzeFrequencyDomain;

beforeAll(async () => {
  const mod = await import('../../lib/detector.js');
  analyzeFrequencyDomain = mod.analyzeFrequencyDomain;
});

describe('L4: FFT — minimum size guard', () => {
  it('returns score 0 for image smaller than 64x64', () => {
    const small = new ImageData(32, 32);
    const result = analyzeFrequencyDomain(small);
    expect(result.score).toBe(0);
    expect(result.signals.length).toBe(0);
  });

  it('processes a 64x64 image without throwing', () => {
    const img = makeRealCameraImageData(64, 64);
    expect(() => analyzeFrequencyDomain(img)).not.toThrow();
  });

  it('processes a 512x512 image without throwing', () => {
    const img = makeRealCameraImageData(512, 512);
    expect(() => analyzeFrequencyDomain(img)).not.toThrow();
  });
});

describe('L4: FFT — random noise image (no SynthID)', () => {
  it('returns low score for pseudo-random noise (no phase-coherent carriers)', () => {
    // Random noise should not have SynthID-like phase coherence
    const img = makeRealCameraImageData(256, 256);
    const result = analyzeFrequencyDomain(img);
    // Random noise won't have coherent carriers — score should be low
    expect(result.score).toBeLessThan(0.5);
  });

  it('returns fft-type signals if any fires', () => {
    const img = makeRealCameraImageData(256, 256);
    const result = analyzeFrequencyDomain(img);
    if (result.signals.length > 0) {
      expect(result.signals.every(s => s.type === 'fft')).toBe(true);
    }
  });
});

describe('L4: FFT — score bounds', () => {
  it('score never exceeds 0.84 (L4 max cap)', () => {
    // Test with multiple image sizes
    for (const size of [64, 128, 256, 512]) {
      const img = makeRealCameraImageData(size, size);
      const result = analyzeFrequencyDomain(img);
      expect(result.score).toBeLessThanOrEqual(0.84);
    }
  });

  it('score is always >= 0', () => {
    const img = makeRealCameraImageData(128, 128);
    const result = analyzeFrequencyDomain(img);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });
});

describe('L4: FFT — non-power-of-2 dimensions', () => {
  it('handles 300x200 image (non-power-of-2) without throwing', () => {
    const img = makeRealCameraImageData(300, 200);
    expect(() => analyzeFrequencyDomain(img)).not.toThrow();
  });

  it('handles 1024x768 image without throwing', () => {
    const img = makeRealCameraImageData(1024, 768);
    expect(() => analyzeFrequencyDomain(img)).not.toThrow();
  });
});
