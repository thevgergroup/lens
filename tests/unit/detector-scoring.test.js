/**
 * Unit tests for score interpretation and result building
 * Tests interpretScore() and CONFIDENCE_THRESHOLDS.
 */

import './helpers/browser-stubs.js';
import { describe, it, expect, beforeAll } from 'vitest';

let interpretScore, CONFIDENCE_THRESHOLDS;

beforeAll(async () => {
  const mod = await import('../../lib/detector.js');
  interpretScore = mod.interpretScore;
  CONFIDENCE_THRESHOLDS = mod.CONFIDENCE_THRESHOLDS;
});

describe('Score interpretation — threshold boundaries', () => {
  it('scores >= 0.90 are "definite"', () => {
    expect(interpretScore(0.90).level).toBe('definite');
    expect(interpretScore(0.95).level).toBe('definite');
    expect(interpretScore(1.00).level).toBe('definite');
  });

  it('scores >= 0.70 and < 0.90 are "likely"', () => {
    expect(interpretScore(0.70).level).toBe('likely');
    expect(interpretScore(0.80).level).toBe('likely');
    expect(interpretScore(0.89).level).toBe('likely');
  });

  it('scores >= 0.45 and < 0.70 are "possible"', () => {
    expect(interpretScore(0.45).level).toBe('possible');
    expect(interpretScore(0.60).level).toBe('possible');
    expect(interpretScore(0.69).level).toBe('possible');
  });

  it('scores >= 0.20 and < 0.45 are "unlikely"', () => {
    expect(interpretScore(0.20).level).toBe('unlikely');
    expect(interpretScore(0.30).level).toBe('unlikely');
    expect(interpretScore(0.44).level).toBe('unlikely');
  });

  it('scores < 0.20 are "clean"', () => {
    expect(interpretScore(0.00).level).toBe('clean');
    expect(interpretScore(0.10).level).toBe('clean');
    expect(interpretScore(0.19).level).toBe('clean');
  });
});

describe('Score interpretation — labels', () => {
  it('"definite" level has label "AI Generated"', () => {
    expect(interpretScore(0.95).label).toBe('AI Generated');
  });

  it('"likely" level has label "Likely AI"', () => {
    expect(interpretScore(0.75).label).toBe('Likely AI');
  });

  it('"possible" level has label "Possible AI"', () => {
    expect(interpretScore(0.50).label).toBe('Possible AI');
  });

  it('"unlikely" level has label "Probably Real"', () => {
    expect(interpretScore(0.30).label).toBe('Probably Real');
  });

  it('"clean" level has label "No AI signals"', () => {
    expect(interpretScore(0.05).label).toBe('No AI signals');
  });
});

describe('CONFIDENCE_THRESHOLDS constants', () => {
  it('DEFINITE is 0.90', () => {
    expect(CONFIDENCE_THRESHOLDS.DEFINITE).toBe(0.90);
  });

  it('LIKELY is 0.70', () => {
    expect(CONFIDENCE_THRESHOLDS.LIKELY).toBe(0.70);
  });

  it('POSSIBLE is 0.45', () => {
    expect(CONFIDENCE_THRESHOLDS.POSSIBLE).toBe(0.45);
  });

  it('UNLIKELY is 0.20', () => {
    expect(CONFIDENCE_THRESHOLDS.UNLIKELY).toBe(0.20);
  });
});
