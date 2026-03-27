/**
 * Unit tests for Layer 1: URL/hostname heuristics
 * Tests checkUrlHeuristics() in isolation.
 */

import './helpers/browser-stubs.js';
import { describe, it, expect } from 'vitest';

// Import only the URL heuristics function
// We extract it by importing the module and using the named export
// Since detector.js doesn't export individually, we test via a thin wrapper
// OR we restructure to allow import. For now we test the full analyzeImage path
// with a mock fetch that returns empty bytes (so only L1 fires).

// detector.js is designed as a module with exported functions
// Let's verify what's exported
let checkUrlHeuristics;

beforeAll(async () => {
  const mod = await import('../../lib/detector.js');
  checkUrlHeuristics = mod.checkUrlHeuristics;
});

describe('L1: URL heuristics — known AI CDN hostnames', () => {
  it('flags oaidalleapiprodscus.blob.core.windows.net (OpenAI CDN)', () => {
    const result = checkUrlHeuristics('https://oaidalleapiprodscus.blob.core.windows.net/private/img/sample.png');
    expect(result.score).toBeGreaterThanOrEqual(0.90);
    expect(result.signals[0].type).toBe('url');
  });

  it('flags cdn.midjourney.com', () => {
    const result = checkUrlHeuristics('https://cdn.midjourney.com/abc123/0_0.png');
    expect(result.score).toBeGreaterThanOrEqual(0.90);
    expect(result.signals[0].type).toBe('url');
  });

  it('flags cdn2.stablediffusionapi.com', () => {
    const result = checkUrlHeuristics('https://cdn2.stablediffusionapi.com/generations/image.jpg');
    expect(result.score).toBeGreaterThanOrEqual(0.90);
    expect(result.signals[0].type).toBe('url');
  });

  it('flags leonardo.ai CDN', () => {
    const result = checkUrlHeuristics('https://cdn.leonardo.ai/users/abc/images/sample.jpg');
    expect(result.score).toBeGreaterThanOrEqual(0.90);
  });
});

describe('L1: URL heuristics — path/query patterns', () => {
  it('flags ?model=sdxl query param', () => {
    const result = checkUrlHeuristics('https://example.com/image.jpg?model=sdxl');
    expect(result.score).toBeGreaterThan(0);
    expect(result.signals.length).toBeGreaterThan(0);
  });

  it('flags /dalle-2/ in path', () => {
    const result = checkUrlHeuristics('https://example.com/dalle-2/output/image.png');
    expect(result.score).toBeGreaterThan(0);
  });

  it('flags /midjourney/ in path', () => {
    const result = checkUrlHeuristics('https://example.com/midjourney/grid/abc.jpg');
    expect(result.score).toBeGreaterThan(0);
  });
});

describe('L1: URL heuristics — filename patterns', () => {
  it('flags filenames starting with DALL-E', () => {
    const result = checkUrlHeuristics('https://example.com/assets/DALL-E-sample-output.png');
    expect(result.score).toBeGreaterThan(0);
  });

  it('flags filenames starting with MJ-', () => {
    const result = checkUrlHeuristics('https://example.com/MJ-abc123def456.png');
    expect(result.score).toBeGreaterThan(0);
  });

  it('flags _generated_ in filename', () => {
    const result = checkUrlHeuristics('https://example.com/article_generated_hero.jpg');
    expect(result.score).toBeGreaterThan(0);
  });
});

describe('L1: URL heuristics — should NOT flag', () => {
  it('does not flag a normal news photo URL', () => {
    const result = checkUrlHeuristics('https://images.reuters.com/news/2026/photo.jpg');
    expect(result.score).toBe(0);
    expect(result.signals.length).toBe(0);
  });

  it('does not flag a Wikipedia image URL', () => {
    const result = checkUrlHeuristics('https://upload.wikimedia.org/wikipedia/commons/thumb/photo.jpg');
    expect(result.score).toBe(0);
  });

  it('returns score 0 for data: URLs', () => {
    const result = checkUrlHeuristics('data:image/png;base64,abc123');
    expect(result.score).toBe(0);
  });

  it('returns score 0 for empty string', () => {
    const result = checkUrlHeuristics('');
    expect(result.score).toBe(0);
  });
});
