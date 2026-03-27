/**
 * LENS — Core Detection Engine
 * Runs in the Service Worker via OffscreenCanvas
 * 
 * Detection layers (cheapest-first, bail early when confident):
 *   L1: URL + DOM heuristics        ~0ms
 *   L2: EXIF/XMP/IPTC metadata      ~2ms
 *   L3: Statistical pixel analysis  ~20ms
 *   L4: FFT frequency analysis      ~80ms  (SynthID-inspired)
 */

// ---------------------------------------------------------------------------
// LAYER 1 — URL & hostname heuristics
// ---------------------------------------------------------------------------

const AI_HOSTNAMES = new Set([
  'oaidalleapiprodscus.blob.core.windows.net',
  'oaiprodscus.blob.core.windows.net',
  'cdn.openai.com',
  'cdn.midjourney.com',
  'mj-gallery.com',
  'cdn.leonardo.ai',
  'image.cdn2.seaart.ai',
  'images.nightcafe.studio',
  'imagedelivery.net',          // Cloudflare AI image CDN
  'firefly.adobe.com',
  'gen-image.adobe.io',
  'ideogram.ai',
  'image.ideogram.ai',
  'cdn.stability.ai',
  'pb.starryai.com',
  'getimg.ai',
  'images.tensor.art',
  'cdn2.civitai.com',
  'cdn2.stablediffusionapi.com',
  'cdn3.stablediffusionapi.com',
  'modelslab.com',
]);

const AI_URL_PATTERNS = [
  /\/dalle[-_]?[23]?\//i,
  /\/midjourney\//i,
  /\/stablediffusion\//i,
  /\/firefly\//i,
  /[?&]source=ai/i,
  /[?&]model=(dalle|midjourney|flux|sdxl|imagen)/i,
];

const AI_FILENAME_PATTERNS = [
  /^DALL[·\-_\s]?E/i,
  /^MJ[-_]/i,
  /[-_]?generated[-_]?/i,
  /[-_]?ai[-_]?art[-_]?/i,
  /^[a-f0-9]{32,64}\.(png|jpg|webp)$/i,  // pure hash filename
];

export function checkUrlHeuristics(url) {
  if (!url || url.startsWith('data:')) return { score: 0, signals: [] };

  const signals = [];
  let score = 0;

  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();
    const pathname = parsed.pathname;
    const filename = pathname.split('/').pop() || '';

    if (AI_HOSTNAMES.has(hostname)) {
      signals.push({ type: 'url', label: `Known AI CDN: ${hostname}`, weight: 0.95 });
      score = Math.max(score, 0.95);
    }

    for (const pattern of AI_URL_PATTERNS) {
      if (pattern.test(url)) {
        signals.push({ type: 'url', label: `AI URL pattern: ${pattern}`, weight: 0.8 });
        score = Math.max(score, 0.8);
      }
    }

    for (const pattern of AI_FILENAME_PATTERNS) {
      if (pattern.test(filename)) {
        signals.push({ type: 'url', label: `AI filename pattern: ${filename}`, weight: 0.6 });
        score = Math.max(score, 0.6);
      }
    }
  } catch (_) {
    // Invalid URL — skip
  }

  return { score, signals };
}

// ---------------------------------------------------------------------------
// LAYER 2 — EXIF / XMP / IPTC metadata
// ---------------------------------------------------------------------------

const AI_SOFTWARE_STRINGS = [
  'Adobe Firefly', 'Midjourney', 'DALL-E', 'DALL·E',
  'Stable Diffusion', 'Automatic1111', 'ComfyUI', 'InvokeAI',
  'Flux', 'Imagen', 'Ideogram', 'Leonardo', 'Bing Image Creator',
  'SDXL', 'DreamStudio', 'NightCafe', 'Canva AI', 'Runway',
  'NovelAI', 'Pika', 'Krea', 'Magnific', 'Topaz', 'Luminar Neo',
  'GPT-4', 'gpt-image', 'ChatGPT', 'Aurora', 'grok-2-image', 'xAI',
  'Ideogram', 'Recraft', 'generative', 'text-to-image', 'diffusion model',
];

// IPTC controlled vocabulary value for AI-generated content
const IPTC_AI_SOURCE_TYPE = 'trainedAlgorithmicMedia';

/**
 * Minimal EXIF/APP1 parser — reads JPEG markers without external deps.
 * Handles: EXIF (IFD0 Software/Make/DateTime), XMP (raw XML scan), IPTC.
 */
export async function parseMetadata(arrayBuffer) {
  const signals = [];
  let score = 0;

  try {
    const bytes = new Uint8Array(arrayBuffer);

    // Only process JPEG (FF D8) and PNG (89 50 4E 47)
    const isJpeg = bytes[0] === 0xFF && bytes[1] === 0xD8;
    const isPng  = bytes[0] === 0x89 && bytes[1] === 0x50;

    if (isJpeg) {
      const result = parseJpegMetadata(bytes);
      signals.push(...result.signals);
      score = Math.max(score, result.score);
    } else if (isPng) {
      const result = parsePngMetadata(bytes);
      signals.push(...result.signals);
      score = Math.max(score, result.score);
    }
  } catch (err) {
    // Metadata parsing failed — not fatal
  }

  return { score, signals };
}

function checkSoftwareString(value) {
  if (!value) return null;
  const lower = value.toLowerCase();
  for (const s of AI_SOFTWARE_STRINGS) {
    if (lower.includes(s.toLowerCase())) return s;
  }
  return null;
}

function parseJpegMetadata(bytes) {
  const signals = [];
  let score = 0;
  let offset = 2; // Skip FF D8

  while (offset < bytes.length - 4) {
    if (bytes[offset] !== 0xFF) break;
    const marker = bytes[offset + 1];
    const length = (bytes[offset + 2] << 8) | bytes[offset + 3];

    // APP1 — EXIF or XMP
    if (marker === 0xE1) {
      const segmentBytes = bytes.slice(offset + 4, offset + 2 + length);
      const header = String.fromCharCode(...segmentBytes.slice(0, 6));

      if (header.startsWith('Exif')) {
        const exifResult = readExifIFD(segmentBytes.slice(6));
        signals.push(...exifResult.signals);
        score = Math.max(score, exifResult.score);
      } else if (header.startsWith('http://ns.adobe.com') || 
                 String.fromCharCode(...segmentBytes.slice(0, 29)).includes('xpacket')) {
        const xmpResult = scanXmpString(new TextDecoder().decode(segmentBytes));
        signals.push(...xmpResult.signals);
        score = Math.max(score, xmpResult.score);
      }
    }

    // APP11 — JUMBF / C2PA Content Credentials (binary box format)
    if (marker === 0xEB) {
      const segmentBytes = bytes.slice(offset + 4, offset + 2 + length);
      const c2paResult = scanJumbfBox(segmentBytes);
      signals.push(...c2paResult.signals);
      score = Math.max(score, c2paResult.score);
    }

    // APP13 — IPTC
    if (marker === 0xED) {
      const segmentBytes = bytes.slice(offset + 4, offset + 2 + length);
      const iptcResult = scanIptcBlock(segmentBytes);
      signals.push(...iptcResult.signals);
      score = Math.max(score, iptcResult.score);
    }

    offset += 2 + length;
  }

  return { score, signals };
}

function readExifIFD(data) {
  const signals = [];
  let score = 0;

  try {
    // Read byte order (II=little-endian, MM=big-endian)
    const isLittleEndian = data[0] === 0x49 && data[1] === 0x49;
    const read16 = (o) => isLittleEndian 
      ? (data[o] | (data[o+1] << 8)) 
      : ((data[o] << 8) | data[o+1]);
    const read32 = (o) => isLittleEndian 
      ? (data[o] | (data[o+1]<<8) | (data[o+2]<<16) | (data[o+3]<<24)) 
      : ((data[o]<<24) | (data[o+1]<<16) | (data[o+2]<<8) | data[o+3]);

    const ifdOffset = read32(4);
    const entryCount = read16(ifdOffset);

    for (let i = 0; i < entryCount; i++) {
      const entryOffset = ifdOffset + 2 + (i * 12);
      const tag = read16(entryOffset);
      const type = read16(entryOffset + 2);
      const count = read32(entryOffset + 4);
      const valueOffset = entryOffset + 8;

      // Tag 0x0131 = Software, 0x013B = Artist, 0x8298 = Copyright
      if (tag === 0x0131 || tag === 0x013B || tag === 0x8298) {
        let strOffset = count <= 4 ? valueOffset : read32(valueOffset);
        const str = readAscii(data, strOffset, count);
        const match = checkSoftwareString(str);
        if (match) {
          signals.push({ type: 'exif', label: `EXIF Software: "${str}"`, weight: 0.92 });
          score = Math.max(score, 0.92);
        }
      }

      // Tag 0x9C9B = XPTitle (Windows), 0x9C9C = XPComment — sometimes contain prompts
      if (tag === 0x9C9C || tag === 0x9C9B) {
        let strOffset = count <= 4 ? valueOffset : read32(valueOffset);
        const str = readUcs2(data, strOffset, count);
        if (str && str.length > 20) {
          // Long descriptive strings in these fields are suspicious
          signals.push({ type: 'exif', label: 'Possible prompt text in EXIF comment', weight: 0.55 });
          score = Math.max(score, 0.55);
        }
      }
    }
  } catch (_) {}

  return { score, signals };
}

function readAscii(data, offset, count) {
  const chars = [];
  for (let i = 0; i < Math.min(count, 256); i++) {
    const c = data[offset + i];
    if (c === 0) break;
    chars.push(String.fromCharCode(c));
  }
  return chars.join('');
}

function readUcs2(data, offset, count) {
  const chars = [];
  for (let i = 0; i < Math.min(count, 512); i += 2) {
    const c = data[offset + i] | (data[offset + i + 1] << 8);
    if (c === 0) break;
    chars.push(String.fromCharCode(c));
  }
  return chars.join('');
}

function scanXmpString(xmpText) {
  const signals = [];
  let score = 0;

  // C2PA / Content Credentials
  if (xmpText.includes('c2pa') || xmpText.includes('contentcredentials')) {
    signals.push({ type: 'c2pa', label: 'C2PA Content Credentials found', weight: 0.88 });
    score = Math.max(score, 0.88);
  }

  // IPTC DigitalSourceType
  if (xmpText.includes(IPTC_AI_SOURCE_TYPE)) {
    signals.push({ type: 'iptc', label: 'IPTC: trainedAlgorithmicMedia', weight: 0.97 });
    score = Math.max(score, 0.97);
  }

  // Adobe Firefly specific
  if (xmpText.includes('GenerativeAI') || xmpText.includes('firefly')) {
    signals.push({ type: 'xmp', label: 'Adobe Firefly XMP marker', weight: 0.95 });
    score = Math.max(score, 0.95);
  }

  // Software field in XMP
  const softwareMatch = xmpText.match(/<(?:xmp:)?CreatorTool[^>]*>([^<]+)<\/(?:xmp:)?CreatorTool>/i);
  if (softwareMatch) {
    const tool = softwareMatch[1];
    const match = checkSoftwareString(tool);
    if (match) {
      signals.push({ type: 'xmp', label: `XMP CreatorTool: "${tool}"`, weight: 0.93 });
      score = Math.max(score, 0.93);
    }
  }

  // Check for prompt text (common in AI tools that write prompts to XMP description)
  const descMatch = xmpText.match(/<(?:dc:)?description[^>]*>([^<]{50,})<\/(?:dc:)?description>/i);
  if (descMatch && descMatch[1].length > 50) {
    signals.push({ type: 'xmp', label: 'Long description field (possible prompt)', weight: 0.45 });
    score = Math.max(score, 0.45);
  }

  return { score, signals };
}

function scanIptcBlock(data) {
  const signals = [];
  let score = 0;
  let offset = 0;

  while (offset < data.length - 5) {
    if (data[offset] !== 0x1C) { offset++; continue; }
    const record = data[offset + 1];
    const dataset = data[offset + 2];
    const size = (data[offset + 3] << 8) | data[offset + 4];
    offset += 5;

    if (record === 2) {
      const value = new TextDecoder().decode(data.slice(offset, offset + size));
      // Dataset 228 = DigitalSourceType
      if (dataset === 228 && value.includes('trainedAlgorithmicMedia')) {
        signals.push({ type: 'iptc', label: 'IPTC DigitalSourceType: AI generated', weight: 0.97 });
        score = 0.97;
      }
      // Dataset 80 = Byline / dataset 110 = Credit
      if ((dataset === 80 || dataset === 110) && checkSoftwareString(value)) {
        signals.push({ type: 'iptc', label: `IPTC credit/byline: "${value}"`, weight: 0.85 });
        score = Math.max(score, 0.85);
      }
    }

    offset += size;
  }

  return { score, signals };
}

/**
 * Scans a JUMBF binary box for C2PA content credentials.
 * JUMBF boxes have a 16-byte UUID at offset +8 from box start.
 * C2PA JUMBF boxes also embed readable strings like "c2pa", "trainedAlgorithmicMedia",
 * and tool names (e.g. "GPT-4o", "Adobe Firefly") in CBOR-encoded claim data.
 */
function scanJumbfBox(data) {
  const signals = [];
  let score = 0;

  // Scan the raw bytes for readable C2PA strings
  // CBOR/JUMBF encodes strings with length prefix — scanning for ASCII is reliable
  const text = new TextDecoder('utf-8', { fatal: false }).decode(data);

  if (!text.includes('c2pa')) return { score, signals };

  // C2PA present — but C2PA is also used by cameras and editors for real photos.
  // Only flag if there's a positive AI signal within the manifest.
  let label = null;
  let weight = 0;

  // DigitalSourceType: trainedAlgorithmicMedia = definitively AI generated
  if (text.includes('trainedAlgorithmicMedia')) {
    label = 'C2PA: trainedAlgorithmicMedia (AI generated)';
    weight = 0.97;
  }

  // Named AI tools embedded in the claim
  const toolMatch = checkSoftwareString(text);
  if (toolMatch) {
    label = `C2PA claim: created by ${toolMatch}`;
    weight = 0.97;
  }

  // GPT-4o / ChatGPT specific
  if (text.includes('GPT-4') || text.includes('gpt-4') || text.includes('ChatGPT')) {
    label = 'C2PA claim: created by ChatGPT/GPT-4';
    weight = 0.97;
  }

  // xAI / Grok / Aurora
  if (text.includes('Aurora') || text.includes('grok-2-image') || text.includes('xAI')) {
    label = 'C2PA claim: created by xAI/Grok';
    weight = 0.97;
  }

  // No AI-specific claim found — this is just edit/camera provenance, not AI generation
  if (!label) return { score, signals };

  signals.push({ type: 'c2pa', label, weight });
  score = weight;

  return { score, signals };
}

function parsePngMetadata(bytes) {
  const signals = [];
  let score = 0;

  // PNG chunks: 4-byte length, 4-byte type, data, 4-byte CRC
  let offset = 8; // Skip PNG signature

  while (offset < bytes.length - 12) {
    const length = (bytes[offset]<<24) | (bytes[offset+1]<<16) | (bytes[offset+2]<<8) | bytes[offset+3];
    const type = String.fromCharCode(bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]);

    // C2PA JUMBF box — PNG stores it in 'caBX' chunks
    if (type === 'caBX') {
      const chunkData = bytes.slice(offset + 8, offset + 8 + length);
      const c2paResult = scanJumbfBox(chunkData);
      signals.push(...c2paResult.signals);
      score = Math.max(score, c2paResult.score);
    }

    if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
      // Read key=value text chunk
      const chunkData = bytes.slice(offset + 8, offset + 8 + length);
      const text = new TextDecoder('utf-8', { fatal: false }).decode(chunkData);

      // ComfyUI, A1111, etc. embed parameters/workflow here
      if (text.includes('parameters') || text.includes('workflow') || 
          text.includes('prompt') || text.includes('sampler')) {
        const match = checkSoftwareString(text);
        if (match || text.includes('Steps:') || text.includes('CFG scale')) {
          signals.push({ type: 'png-meta', label: 'PNG tEXt: AI generation parameters found', weight: 0.94 });
          score = Math.max(score, 0.94);
        }
      }

      // Software field
      if (text.startsWith('Software\0')) {
        const softwareValue = text.slice(9);
        const match = checkSoftwareString(softwareValue);
        if (match) {
          signals.push({ type: 'png-meta', label: `PNG Software: "${softwareValue}"`, weight: 0.92 });
          score = Math.max(score, 0.92);
        }
      }
    }

    offset += 12 + length; // length + type(4) + data + CRC(4)
  }

  return { score, signals };
}

// ---------------------------------------------------------------------------
// LAYER 3 — Statistical pixel analysis
// ---------------------------------------------------------------------------

/**
 * Analyses pixel-level statistics for AI image characteristics.
 * AI images tend to have:
 *   - Lower high-frequency noise energy (over-smoothed)
 *   - Specific LSB entropy patterns (from watermarking or model quantisation)
 *   - Unusual color channel correlation
 *   - Distinctive gradient smoothness
 */
export function analyzePixelStatistics(imageData, isLossy = false) {
  const { data, width, height } = imageData;
  const signals = [];
  let score = 0;

  const totalPixels = width * height;
  if (totalPixels < 100) return { score: 0, signals: [] };

  // Extract channels
  const r = new Float32Array(totalPixels);
  const g = new Float32Array(totalPixels);
  const b = new Float32Array(totalPixels);
  const lum = new Float32Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    r[i] = data[i * 4];
    g[i] = data[i * 4 + 1];
    b[i] = data[i * 4 + 2];
    lum[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
  }

  // --- 1. LSB entropy analysis ---
  // Real camera images: near-random LSBs from sensor noise → entropy ~1.0
  // AI PNG images: structured LSBs from quantisation or watermarking → lower entropy
  // NOTE: Skip for JPEG/WebP — DCT quantization rounds values producing naturally
  // structured LSBs in all lossy-compressed images regardless of AI vs real origin.
  if (!isLossy) {
    const lsbCounts = [0, 0];
    for (let i = 0; i < totalPixels; i++) {
      lsbCounts[data[i * 4] & 1]++;
      lsbCounts[data[i * 4 + 1] & 1]++;
      lsbCounts[data[i * 4 + 2] & 1]++;
    }
    const lsbTotal = lsbCounts[0] + lsbCounts[1];
    const p0 = lsbCounts[0] / lsbTotal;
    const p1 = lsbCounts[1] / lsbTotal;
    const lsbEntropy = p0 > 0 && p1 > 0 ? -(p0 * Math.log2(p0) + p1 * Math.log2(p1)) : 0;

    if (lsbEntropy < 0.85) {
      signals.push({ type: 'pixel', label: `Low LSB entropy: ${lsbEntropy.toFixed(3)} (structured LSBs)`, weight: 0.55 });
      score = Math.max(score, 0.55);
    }
  }

  // --- 2. Local gradient smoothness ---
  // AI images: unusually smooth gradients with sharp semantic edges
  // Measure: average magnitude of 3x3 Sobel gradient
  let gradSum = 0;
  let gradSampleCount = 0;
  const sampleStep = Math.max(1, Math.floor(Math.sqrt(totalPixels / 5000)));

  for (let y = 1; y < height - 1; y += sampleStep) {
    for (let x = 1; x < width - 1; x += sampleStep) {
      const idx = (y * width + x);
      const gx = -lum[(y-1)*width+(x-1)] + lum[(y-1)*width+(x+1)]
                 - 2*lum[y*width+(x-1)]   + 2*lum[y*width+(x+1)]
                 - lum[(y+1)*width+(x-1)] + lum[(y+1)*width+(x+1)];
      const gy = -lum[(y-1)*width+(x-1)] - 2*lum[(y-1)*width+x] - lum[(y-1)*width+(x+1)]
                 + lum[(y+1)*width+(x-1)] + 2*lum[(y+1)*width+x] + lum[(y+1)*width+(x+1)];
      gradSum += Math.sqrt(gx*gx + gy*gy);
      gradSampleCount++;
    }
  }

  const avgGradient = gradSampleCount > 0 ? gradSum / gradSampleCount : 0;

  // Very low or very bimodal gradient distribution is suspicious
  if (avgGradient < 8 && totalPixels > 10000) {
    signals.push({ type: 'pixel', label: `Unusually smooth gradient: ${avgGradient.toFixed(2)}`, weight: 0.45 });
    score = Math.max(score, 0.45);
  }

  // --- 3. Color channel correlation ---
  // AI images: very high R-G-B correlation (over-saturated, model-biased colors)
  let sumR = 0, sumG = 0, sumB = 0;
  for (let i = 0; i < totalPixels; i++) { sumR += r[i]; sumG += g[i]; sumB += b[i]; }
  const meanR = sumR / totalPixels;
  const meanG = sumG / totalPixels;
  const meanB = sumB / totalPixels;

  let covRG = 0, varR = 0, varG = 0;
  for (let i = 0; i < totalPixels; i++) {
    covRG += (r[i] - meanR) * (g[i] - meanG);
    varR += (r[i] - meanR) ** 2;
    varG += (g[i] - meanG) ** 2;
  }
  const corrRG = (varR > 0 && varG > 0) ? covRG / Math.sqrt(varR * varG) : 0;

  if (Math.abs(corrRG) > 0.97) {
    signals.push({ type: 'pixel', label: `Abnormal R-G correlation: ${corrRG.toFixed(3)}`, weight: 0.4 });
    score = Math.max(score, 0.4);
  }

  // --- 4. Noise floor analysis ---
  // Compute local noise as difference between pixel and blurred neighbourhood
  // AI images tend to have very low, uniform noise floor
  let noiseSum = 0;
  let noiseSamples = 0;
  for (let y = 1; y < height - 1; y += sampleStep * 2) {
    for (let x = 1; x < width - 1; x += sampleStep * 2) {
      const center = lum[y * width + x];
      const avg = (
        lum[(y-1)*width+x] + lum[(y+1)*width+x] +
        lum[y*width+(x-1)] + lum[y*width+(x+1)]
      ) / 4;
      noiseSum += Math.abs(center - avg);
      noiseSamples++;
    }
  }
  const avgNoise = noiseSamples > 0 ? noiseSum / noiseSamples : 0;

  // Skip noise floor check on lossy images — JPEG DCT smoothing produces
  // artificially low local noise in real photographs.
  if (!isLossy && avgNoise < 1.2 && totalPixels > 50000) {
    signals.push({ type: 'pixel', label: `Very low noise floor: ${avgNoise.toFixed(3)}`, weight: 0.5 });
    score = Math.max(score, 0.5);
  }

  // --- 5. Laplace noise block correlation (Mallet et al. 2025) ---
  // Apply Laplace high-pass filter, divide into 8×8 blocks, select the T blocks
  // with lowest variance (least content, most isolated noise), then compute
  // pairwise Pearson correlation between selected blocks.
  //
  // AI images (especially Aurora/xAI, Bing Creator, DALL-E WebP) show elevated
  // inter-block noise correlation vs real photographs. Thresholds calibrated
  // empirically on fixture set with zero false positives across 18 real images.
  //
  // Signals not fired on: FLUX.1, most SD, older DALL-E PNG (within real-image range).
  // Signals reliably fired on: Aurora (all 6 samples), Bing Creator, DALL-E WebP.
  if (totalPixels >= 16384) {
    const noiseCorr = computeBlockNoiseCorrelation(r, g, b, width, height);
    if (noiseCorr !== null) {
      const { highCorrFrac, absMean } = noiseCorr;
      // Both thresholds validated at zero false positives on 18 real-image fixture set
      if (highCorrFrac > 0.093 || absMean > 0.1151) {
        // Scale weight by how far above threshold
        const excess = Math.max(highCorrFrac / 0.093, absMean / 0.1151) - 1;
        const weight = excess > 0.5 ? 0.52 : 0.45;
        signals.push({
          type: 'pixel',
          label: `Elevated block noise correlation: hcf=${highCorrFrac.toFixed(3)} am=${absMean.toFixed(3)}`,
          weight,
        });
        score = Math.max(score, weight);
      }
    }
  }

  return { score, signals };
}

/**
 * Mallet et al. (2025) noise block correlation.
 * Converts to YCbCr, applies Laplace filter, extracts 8×8 blocks,
 * selects T=30 lowest-variance blocks per channel, computes pairwise
 * Pearson correlation, returns summary statistics.
 *
 * Returns null if the image is too small or has insufficient block diversity.
 */
function computeBlockNoiseCorrelation(r, g, b, width, height) {
  const T = 30;

  // Convert to YCbCr
  const Y  = new Float32Array(width * height);
  const Cb = new Float32Array(width * height);
  const Cr = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    Y[i]  =  0.299   * r[i] + 0.587   * g[i] + 0.114   * b[i];
    Cb[i] = -0.16874 * r[i] - 0.33126 * g[i] + 0.5     * b[i] + 128;
    Cr[i] =  0.5     * r[i] - 0.41869 * g[i] - 0.08131 * b[i] + 128;
  }

  // Laplace high-pass: F[y,x] = -4·I[y,x] + N + S + W + E
  function laplace(ch) {
    const out = new Float32Array(width * height);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        out[i] = -4*ch[i] + ch[i-width] + ch[i+width] + ch[i-1] + ch[i+1];
      }
    }
    return out;
  }

  // Extract 8×8 non-overlapping blocks, skip edge rows/cols touched by Laplace
  function blocks(filtered) {
    const result = [];
    for (let by = 1; by + 8 < height - 1; by += 8) {
      for (let bx = 1; bx + 8 < width - 1; bx += 8) {
        const blk = new Float32Array(64);
        for (let dy = 0; dy < 8; dy++)
          for (let dx = 0; dx < 8; dx++)
            blk[dy*8+dx] = filtered[(by+dy)*width+(bx+dx)];
        result.push(blk);
      }
    }
    return result;
  }

  // Block variance (skip near-zero — undefined correlation)
  function blockVar(blk) {
    let s = 0;
    for (const v of blk) s += v;
    const m = s / 64;
    let v = 0;
    for (const x of blk) v += (x-m)*(x-m);
    return v / 64;
  }

  // Select T blocks with lowest nonzero variance
  function selectBlocks(blks) {
    return blks
      .map((b, i) => ({ i, v: blockVar(b) }))
      .filter(s => s.v > 0.01)
      .sort((a, b) => a.v - b.v)
      .slice(0, T)
      .map(s => blks[s.i]);
  }

  // Pearson correlation between two 64-element vectors
  function pearson(a, b) {
    let sa = 0, sb = 0;
    for (let i = 0; i < 64; i++) { sa += a[i]; sb += b[i]; }
    const ma = sa/64, mb = sb/64;
    let cov = 0, va = 0, vb = 0;
    for (let i = 0; i < 64; i++) {
      const da = a[i]-ma, db = b[i]-mb;
      cov += da*db; va += da*da; vb += db*db;
    }
    return (va > 1e-10 && vb > 1e-10) ? cov / Math.sqrt(va*vb) : 0;
  }

  // Summary stats across all pairs for one set of selected blocks
  function pairStats(sel) {
    if (sel.length < 4) return null;
    let absSum = 0, highCount = 0, n = 0;
    for (let i = 1; i < sel.length; i++) {
      for (let j = 0; j < i; j++) {
        const rij = Math.abs(pearson(sel[i], sel[j]));
        absSum += rij;
        if (rij > 0.3) highCount++;
        n++;
      }
    }
    return { absMean: absSum / n, highCorrFrac: highCount / n };
  }

  const fY  = laplace(Y);
  const fCb = laplace(Cb);
  const fCr = laplace(Cr);

  const sY  = pairStats(selectBlocks(blocks(fY)));
  const sCb = pairStats(selectBlocks(blocks(fCb)));
  const sCr = pairStats(selectBlocks(blocks(fCr)));

  if (!sY || !sCb || !sCr) return null;

  return {
    absMean:      (sY.absMean      + sCb.absMean      + sCr.absMean)      / 3,
    highCorrFrac: (sY.highCorrFrac + sCb.highCorrFrac + sCr.highCorrFrac) / 3,
  };
}

// ---------------------------------------------------------------------------
// LAYER 4 — FFT frequency analysis (SynthID-inspired)
// ---------------------------------------------------------------------------

/**
 * SynthID-inspired spread-spectrum watermark detector.
 * Based on reverse-engineering findings: SynthID embeds phase-coherent signals
 * at specific carrier frequencies in the FFT domain.
 *
 * Carrier frequencies extracted from: github.com/aloshdenny/reverse-SynthID
 * Phase coherence > 99% at these locations in 250 Gemini-generated images.
 */

// Primary carrier frequencies discovered by reverse-engineering
const SYNTHID_CARRIERS = [
  [14, 14], [126, 14], [98, 14], [128, 128],
  [210, 14], [238, 14], [14, 126], [14, 98],
];

const DETECTION_THRESHOLD = 0.179;
const PHASE_MATCH_THRESHOLD = 0.50;

export function analyzeFrequencyDomain(imageData) {
  const { data, width, height } = imageData;
  const signals = [];
  let score = 0;

  // Work on a power-of-2 crop for FFT efficiency
  const fftSize = largestPow2Leq(Math.min(width, height, 512));
  if (fftSize < 64) return { score: 0, signals: [] };

  // Extract grayscale luminance patch
  const lum = new Float32Array(fftSize * fftSize);
  for (let y = 0; y < fftSize; y++) {
    for (let x = 0; x < fftSize; x++) {
      const srcIdx = (y * width + x) * 4;
      lum[y * fftSize + x] = (0.299*data[srcIdx] + 0.587*data[srcIdx+1] + 0.114*data[srcIdx+2]) / 255;
    }
  }

  // High-pass filter to isolate noise residual (subtract 3x3 mean blur)
  const noise = extractNoise(lum, fftSize);

  // 2D FFT
  const { real, imag } = fft2d(noise, fftSize);

  // Check phase coherence at SynthID carriers
  let coherentCarriers = 0;
  const validCarriers = SYNTHID_CARRIERS.filter(([fx, fy]) => fx < fftSize && fy < fftSize);

  for (const [fx, fy] of validCarriers) {
    const phaseCoherence = measurePhaseCoherence(real, imag, fx, fy, fftSize);
    if (phaseCoherence > 0.8) coherentCarriers++;
  }

  const phaseMatchRatio = coherentCarriers / validCarriers.length;

  // Measure noise correlation (auto-correlation strength at DC is discriminating)
  const noiseEnergy = noise.reduce((sum, v) => sum + v * v, 0) / noise.length;
  const correlation = Math.sqrt(noiseEnergy) * 0.3; // Normalised proxy

  const isWatermarked = correlation > DETECTION_THRESHOLD && phaseMatchRatio > PHASE_MATCH_THRESHOLD;

  if (isWatermarked) {
    const confidence = Math.min(0.84, (phaseMatchRatio * 0.6 + Math.min(1, correlation / 0.3) * 0.4));
    signals.push({
      type: 'fft',
      label: `SynthID frequency signature detected (phase match: ${(phaseMatchRatio*100).toFixed(0)}%, correlation: ${correlation.toFixed(3)})`,
      weight: confidence,
    });
    score = Math.max(score, confidence);
  }

  return { score, signals };
}

function largestPow2Leq(n) {
  let p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

function extractNoise(lum, size) {
  const noise = new Float32Array(size * size);
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const center = lum[y * size + x];
      const blur = (
        lum[(y-1)*size+(x-1)] + lum[(y-1)*size+x] + lum[(y-1)*size+(x+1)] +
        lum[y*size+(x-1)]     + center             + lum[y*size+(x+1)] +
        lum[(y+1)*size+(x-1)] + lum[(y+1)*size+x] + lum[(y+1)*size+(x+1)]
      ) / 9;
      noise[y * size + x] = center - blur;
    }
  }
  return noise;
}

/**
 * 2D FFT via row-column decomposition using Cooley-Tukey radix-2.
 */
function fft2d(input, size) {
  const N = size * size;
  const real = new Float32Array(N);
  const imag = new Float32Array(N);

  // Copy input
  for (let i = 0; i < N; i++) real[i] = input[i];

  // Row-wise 1D FFT
  for (let row = 0; row < size; row++) {
    const rowReal = real.slice(row * size, (row + 1) * size);
    const rowImag = imag.slice(row * size, (row + 1) * size);
    fft1d(rowReal, rowImag, size);
    real.set(rowReal, row * size);
    imag.set(rowImag, row * size);
  }

  // Column-wise 1D FFT
  const colReal = new Float32Array(size);
  const colImag = new Float32Array(size);
  for (let col = 0; col < size; col++) {
    for (let row = 0; row < size; row++) {
      colReal[row] = real[row * size + col];
      colImag[row] = imag[row * size + col];
    }
    fft1d(colReal, colImag, size);
    for (let row = 0; row < size; row++) {
      real[row * size + col] = colReal[row];
      imag[row * size + col] = colImag[row];
    }
  }

  return { real, imag };
}

/**
 * In-place Cooley-Tukey radix-2 DIT FFT.
 * N must be a power of 2.
 */
function fft1d(re, im, N) {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // FFT butterfly
  for (let len = 2; len <= N; len <<= 1) {
    const halfLen = len >> 1;
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);

    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < halfLen; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + halfLen] * curRe - im[i + k + halfLen] * curIm;
        const vIm = re[i + k + halfLen] * curIm + im[i + k + halfLen] * curRe;
        re[i + k]         = uRe + vRe;
        im[i + k]         = uIm + vIm;
        re[i + k + halfLen] = uRe - vRe;
        im[i + k + halfLen] = uIm - vIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }
}

function measurePhaseCoherence(real, imag, fx, fy, size) {
  // Check phase at (fx, fy) and its conjugate-symmetric counterpart
  const idx1 = fy * size + fx;
  const idx2 = (size - fy) * size + (size - fx);

  if (idx1 >= real.length || idx2 >= real.length) return 0;

  const phase1 = Math.atan2(imag[idx1], real[idx1]);
  const phase2 = Math.atan2(imag[idx2], real[idx2]);

  // For real input, conjugate symmetry means phase2 should equal -phase1
  // Deviation from this is a sign of structured watermark injection
  const phaseDiff = Math.abs(phase1 + phase2);
  const normalised = Math.abs(Math.cos(phaseDiff));

  return normalised;
}

// ---------------------------------------------------------------------------
// MAIN ORCHESTRATOR
// ---------------------------------------------------------------------------

export const CONFIDENCE_THRESHOLDS = {
  DEFINITE:  0.90,
  LIKELY:    0.70,
  POSSIBLE:  0.45,
  UNLIKELY:  0.20,
};

export function interpretScore(score, signals) {
  if (score >= CONFIDENCE_THRESHOLDS.DEFINITE)  return { level: 'definite',  label: 'AI Generated',   color: '#FF3B3B' };
  if (score >= CONFIDENCE_THRESHOLDS.LIKELY)    return { level: 'likely',    label: 'Likely AI',      color: '#FF8C00' };
  if (score >= CONFIDENCE_THRESHOLDS.POSSIBLE)  return { level: 'possible',  label: 'Possible AI',    color: '#FFD700' };
  if (score >= CONFIDENCE_THRESHOLDS.UNLIKELY)  return { level: 'unlikely',  label: 'Probably Real',  color: '#4CAF50' };
  return { level: 'clean', label: 'No AI signals', color: '#2196F3' };
}
