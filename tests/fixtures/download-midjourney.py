#!/usr/bin/env python3
"""
Download Midjourney images from ehristoforu/midjourney-images (HuggingFace).

Images are JPEG, mixed sizes — ~40% are below 512px (Discord thumbnails),
so filtering is required. Saves up to MAX_IMAGES to:
  tests/fixtures/images/ai/midjourney/

Usage:
  python3 tests/fixtures/download-midjourney.py
  python3 tests/fixtures/download-midjourney.py --max 400

Requires: datasets, Pillow
  pip install datasets Pillow
"""

import argparse
import random
from pathlib import Path

HF_REPO = "ehristoforu/midjourney-images"
OUT_DIR = Path(__file__).parent / "images" / "ai" / "midjourney"
MIN_SIZE_PX = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=400, help="Max images to download")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        from datasets import load_dataset
        from PIL import Image  # noqa
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install datasets Pillow")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.jpg")))
    needed = args.max - existing
    if needed <= 0:
        print(f"Already have {existing} images, nothing to do.")
        return

    print(f"Streaming {HF_REPO}...")
    print(f"Need {needed} more images (have {existing}, target {args.max})")
    print(f"Note: ~40% of dataset is below {MIN_SIZE_PX}px — will be filtered out")

    ds = load_dataset(HF_REPO, split="train", streaming=True)

    saved = 0
    skipped_size = 0
    skipped_missing = 0
    for row in ds:
        if saved >= needed:
            break
        img = row.get("image")
        if img is None:
            skipped_missing += 1
            continue
        w, h = img.size
        if min(w, h) < MIN_SIZE_PX:
            skipped_size += 1
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = OUT_DIR / f"midjourney_{existing + saved:04d}.jpg"
        img.save(fname, "JPEG", quality=92)
        saved += 1
        if saved % 50 == 0:
            print(f"  Saved {saved}/{needed}...")

    print(f"\n── Done ──")
    print(f"  Saved: {saved} new images")
    print(f"  Skipped (too small): {skipped_size}")
    print(f"  Skipped (missing): {skipped_missing}")
    print(f"  Total: {existing + saved}")
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
