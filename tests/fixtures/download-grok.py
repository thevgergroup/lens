#!/usr/bin/env python3
"""
Downloads Aurora/Grok-generated images for L3/L4 calibration testing.

Two sources:
  1. Known Aurora images served directly from grok.com (retains C2PA metadata)
  2. Known Aurora images that went through X/Twitter CDN (pbs.twimg.com strips metadata)

The CDN-stripped images are the hard case — no metadata survives, so detection
must rely entirely on pixel statistics. These are essential for calibrating L3/L4.

Usage:
  python3 tests/fixtures/download-grok.py

To add new CDN images: find a Grok-generated post on X, right-click the image,
copy the pbs.twimg.com URL (use ?format=jpg&name=large for full resolution),
and add it to TWIMG_URLS below.
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

DEST_AI    = Path(__file__).parent / "images" / "ai" / "grok"
DEST_REAL  = Path(__file__).parent / "images" / "real" / "twitter"

# ---------------------------------------------------------------------------
# Aurora images served through X's CDN (pbs.twimg.com).
# These are confirmed Grok-generated images whose authors posted them on X.
# X strips all EXIF/XMP/C2PA on ingest — pixel analysis only.
# Add more as you find them: right-click → copy image address on X.
# ---------------------------------------------------------------------------
TWIMG_URLS = [
    # Format: (filename, url, credit/source note)
    # Sourced from public X posts where the author stated the image was Grok-generated.
    # These are AI-generated images with no copyright claim.
]

# ---------------------------------------------------------------------------
# Real photographs served through X's CDN (pbs.twimg.com).
# Used to calibrate false positive rate: real photos that also lack metadata
# (because X strips it) should score < 0.20.
# ---------------------------------------------------------------------------
REAL_TWIMG_URLS = [
    # Format: (filename, url, credit/source note)
    # Sourced from public X posts of real photography (news photographers, NASA, etc.)
]

# ---------------------------------------------------------------------------
# Aurora images downloaded directly (retains C2PA / EXIF).
# These validate L2 xAI metadata parsing. The aurora-*.webp images already in
# tests/fixtures/images/ai/ were downloaded this way — don't duplicate them here.
# ---------------------------------------------------------------------------
DIRECT_AURORA_URLS = [
    # Add any new direct grok.com downloads here
]


def download(url, dest, label):
    if dest.exists():
        print(f"  ✓ {dest.name} (cached)")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        print(f"  ✓ {dest.name}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  ✗ {dest.name}: HTTP {e.code} — {label}")
        return False
    except Exception as e:
        print(f"  ✗ {dest.name}: {e} — {label}")
        return False


def main():
    DEST_AI.mkdir(parents=True, exist_ok=True)
    DEST_REAL.mkdir(parents=True, exist_ok=True)

    ok = err = 0

    if TWIMG_URLS:
        print(f"\nDownloading Aurora images via X CDN → {DEST_AI}")
        for fname, url, note in TWIMG_URLS:
            result = download(url, DEST_AI / fname, note)
            if result: ok += 1
            else: err += 1
    else:
        print("\nNo CDN Aurora URLs configured yet.")
        print("To add images: find a Grok-generated post on X, right-click the image,")
        print("copy the pbs.twimg.com URL, and add it to TWIMG_URLS in this script.")

    if REAL_TWIMG_URLS:
        print(f"\nDownloading real photos via X CDN → {DEST_REAL}")
        for fname, url, note in REAL_TWIMG_URLS:
            result = download(url, DEST_REAL / fname, note)
            if result: ok += 1
            else: err += 1
    else:
        print("\nNo real-photo CDN URLs configured yet.")
        print("Add real photography pbs.twimg.com URLs to REAL_TWIMG_URLS for false-positive calibration.")

    if DIRECT_AURORA_URLS:
        print(f"\nDownloading direct Aurora images → {DEST_AI}")
        for fname, url, note in DIRECT_AURORA_URLS:
            result = download(url, DEST_AI / fname, note)
            if result: ok += 1
            else: err += 1

    print(f"\nDone: {ok} downloaded, {err} failed")
    print(f"\nExisting Aurora fixtures: {len(list(DEST_AI.glob('*')))}")
    print(f"Existing real-Twitter fixtures: {len(list(DEST_REAL.glob('*')))}")

    # Remind about the measure scripts for calibration
    print("\nNext steps:")
    print("  node tests/fixtures/measure-noise-signals.mjs  # run L3 signals on all fixtures")
    print("  node tests/fixtures/scan-all.mjs               # run full detector on fixtures")


if __name__ == "__main__":
    main()
