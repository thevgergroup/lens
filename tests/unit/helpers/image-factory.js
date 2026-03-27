/**
 * Generates synthetic ImageData and ArrayBuffers with known properties
 * for testing each detection layer without needing real image files.
 */

/**
 * Creates ImageData filled with uniform noise (simulates real camera).
 * Real cameras have random LSBs → high entropy.
 */
export function makeRealCameraImageData(width = 128, height = 128) {
  const data = new Uint8ClampedArray(width * height * 4);
  // Fill with pseudo-random values using a simple LCG for repeatability
  let seed = 0xdeadbeef;
  for (let i = 0; i < data.length; i++) {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    data[i] = (seed >>> 24) & 0xff;
  }
  // Set alpha to 255
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  return new ImageData(data, width, height);
}

/**
 * Creates ImageData simulating an AI-generated image.
 * Specifically designed to trigger L3 heuristics:
 *
 * - LSB entropy < 0.85: structured LSBs (every pixel LSB = 0)
 * - Gradient < 8 AND totalPixels > 10000: very smooth (constant color regions)
 * - |corrRG| > 0.97: R and G channels nearly identical
 * - avgNoise < 1.2 AND totalPixels > 50000: very low local noise
 *
 * Use width=256, height=256 (65536 px) to trigger the noise floor check.
 */
export function makeAIImageData(width = 256, height = 256) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      // Almost constant color (tiny variation → very low gradient AND low noise)
      // Use a very gentle slope so gradient << 8
      const base = 128 + Math.floor((x / width) * 4); // range 128–132
      // R and G nearly identical → |corrRG| > 0.97
      data[idx] = base;
      data[idx + 1] = base;       // G = R → perfect correlation
      data[idx + 2] = base - 1;
      data[idx + 3] = 255;
      // Clear LSBs to force structured LSB entropy (< 0.85)
      data[idx] &= 0xfe;
      data[idx + 1] &= 0xfe;
      data[idx + 2] &= 0xfe;
    }
  }
  return new ImageData(data, width, height);
}

/**
 * Creates a minimal valid JPEG ArrayBuffer with no EXIF metadata.
 * SOI + minimal APP0 JFIF marker + EOI only — no image data.
 */
export function makeMinimalJpegBuffer() {
  return new Uint8Array([
    0xff, 0xd8,             // SOI
    0xff, 0xe0,             // APP0 marker
    0x00, 0x10,             // length = 16
    0x4a, 0x46, 0x49, 0x46, 0x00, // "JFIF\0"
    0x01, 0x01,             // version 1.1
    0x00,                   // aspect ratio units = 0
    0x00, 0x01,             // X density
    0x00, 0x01,             // Y density
    0x00, 0x00,             // no thumbnail
    0xff, 0xd9,             // EOI
  ]).buffer;
}

/**
 * Creates a JPEG buffer with EXIF APP1 segment containing a Software tag
 * identifying an AI tool.
 *
 * @param {string} softwareName - e.g. 'Adobe Firefly', 'Midjourney', 'DALL-E'
 */
export function makeJpegWithExifSoftware(softwareName) {
  const encoder = new TextEncoder();
  const softwareBytes = encoder.encode(softwareName + '\0');

  // EXIF IFD0: one entry for tag 0x0131 (Software)
  // Little-endian TIFF header
  const ifdEntryCount = 1;
  const ifdEntrySize = 12;
  const tiffHeaderSize = 8;
  const ifdCountSize = 2;
  const ifdNextOffset = 4;
  const valueOffset = tiffHeaderSize + ifdCountSize + (ifdEntryCount * ifdEntrySize) + ifdNextOffset;

  const exifData = new Uint8Array(
    tiffHeaderSize + ifdCountSize + (ifdEntryCount * ifdEntrySize) + ifdNextOffset + softwareBytes.length
  );
  const view = new DataView(exifData.buffer);

  // TIFF header: little-endian, magic 42, IFD offset = 8
  exifData[0] = 0x49; exifData[1] = 0x49; // 'II' = little-endian
  view.setUint16(2, 42, true);             // magic
  view.setUint32(4, 8, true);              // IFD0 offset

  // IFD entry count
  view.setUint16(8, ifdEntryCount, true);

  // IFD entry for Software (tag 0x0131 = 305)
  const entryOffset = 10;
  view.setUint16(entryOffset, 0x0131, true);     // tag
  view.setUint16(entryOffset + 2, 2, true);      // type = ASCII
  view.setUint32(entryOffset + 4, softwareBytes.length, true); // count
  view.setUint32(entryOffset + 8, valueOffset, true);          // value offset

  // Next IFD offset = 0 (no more IFDs)
  view.setUint32(entryOffset + ifdEntrySize, 0, true);

  // Software string value
  exifData.set(softwareBytes, valueOffset);

  // Build APP1 segment: "Exif\0\0" + TIFF data
  const exifMarker = new Uint8Array(6 + exifData.length);
  exifMarker[0] = 0x45; exifMarker[1] = 0x78; exifMarker[2] = 0x69;
  exifMarker[3] = 0x66; exifMarker[4] = 0x00; exifMarker[5] = 0x00; // "Exif\0\0"
  exifMarker.set(exifData, 6);

  const app1Length = 2 + exifMarker.length;
  const jpeg = new Uint8Array(2 + 2 + 2 + exifMarker.length + 2);
  let pos = 0;
  jpeg[pos++] = 0xff; jpeg[pos++] = 0xd8;   // SOI
  jpeg[pos++] = 0xff; jpeg[pos++] = 0xe1;   // APP1 marker
  jpeg[pos++] = (app1Length >> 8) & 0xff;
  jpeg[pos++] = app1Length & 0xff;
  jpeg.set(exifMarker, pos); pos += exifMarker.length;
  jpeg[pos++] = 0xff; jpeg[pos++] = 0xd9;   // EOI

  return jpeg.buffer;
}

/**
 * Creates a PNG buffer with tEXt chunk containing AI generation parameters
 * (as written by AUTOMATIC1111/ComfyUI).
 */
export function makePngWithAIParameters(prompt = 'a beautiful landscape') {
  const encoder = new TextEncoder();

  // PNG signature
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

  // IHDR chunk (1x1 pixel, 8-bit RGB)
  const ihdrData = new Uint8Array([
    0, 0, 0, 1,   // width = 1
    0, 0, 0, 1,   // height = 1
    8,            // bit depth
    2,            // color type = RGB
    0, 0, 0,      // compression, filter, interlace
  ]);
  const ihdr = makePngChunk('IHDR', ihdrData);

  // tEXt chunk: keyword "parameters" + \0 + value (SD-style)
  const keyword = encoder.encode('parameters');
  const value = encoder.encode(
    `${prompt}\nSteps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 42, Model: v1-5`
  );
  const textData = new Uint8Array(keyword.length + 1 + value.length);
  textData.set(keyword, 0);
  textData[keyword.length] = 0; // null separator
  textData.set(value, keyword.length + 1);
  const text = makePngChunk('tEXt', textData);

  // Minimal IDAT (1 pixel, RGB 128,128,128 with filter byte 0)
  // In a real PNG this would be zlib compressed, but for metadata tests we don't need valid IDAT
  const idatData = new Uint8Array([0, 128, 128, 128]);
  const idat = makePngChunk('IDAT', idatData);

  // IEND
  const iend = makePngChunk('IEND', new Uint8Array(0));

  const total = sig.length + ihdr.length + text.length + idat.length + iend.length;
  const buf = new Uint8Array(total);
  let off = 0;
  for (const chunk of [sig, ihdr, text, idat, iend]) {
    buf.set(chunk, off);
    off += chunk.length;
  }
  return buf.buffer;
}

function makePngChunk(type, data) {
  const typeBytes = new TextEncoder().encode(type);
  const chunk = new Uint8Array(4 + 4 + data.length + 4);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length, false);  // length (big-endian)
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);
  // CRC — simplified: use 0 for test purposes (metadata parser doesn't validate CRC)
  view.setUint32(8 + data.length, 0, false);
  return chunk;
}
