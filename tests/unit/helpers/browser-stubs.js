/**
 * Minimal browser API stubs so detector.js can be imported in Node/Vitest.
 * Only stubs what's actually used — no more.
 */

import { TextDecoder as NodeTextDecoder } from 'util';
import { createInflateRaw, inflateRawSync } from 'zlib';

// TextDecoder is available in Node 18+ but may not be global in all test envs
if (typeof globalThis.TextDecoder === 'undefined') {
  globalThis.TextDecoder = NodeTextDecoder;
}

// ImageData — used by pixel analysis layers
if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(dataOrWidth, widthOrHeight, height) {
      if (dataOrWidth instanceof Uint8ClampedArray) {
        this.data = dataOrWidth;
        this.width = widthOrHeight;
        this.height = height ?? (dataOrWidth.length / 4 / widthOrHeight);
      } else {
        // ImageData(width, height)
        this.width = dataOrWidth;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(dataOrWidth * widthOrHeight * 4);
      }
    }
  };
}

// DecompressionStream — used by parseMetadata for zTXt PNG chunks
// Node 18+ has this natively; stub for older environments
if (typeof globalThis.DecompressionStream === 'undefined') {
  globalThis.DecompressionStream = class DecompressionStream {
    constructor(format) {
      this._format = format;
      let controller;
      this.readable = new ReadableStream({
        start(c) { controller = c; },
      });
      this.writable = new WritableStream({
        write: (chunk) => {
          try {
            const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
            const result = inflateRawSync(buf);
            controller.enqueue(new Uint8Array(result));
          } catch {
            // ignore decompression errors in tests — partial data is OK
          }
        },
        close: () => controller.close(),
      });
    }
  };
}
