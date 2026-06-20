#!/usr/bin/env python3
"""
Download DALL-E 3 images from OpenDatasets/dalle-3-dataset (HuggingFace).

Images are 1024px PNG, Discord-sourced DALL-E 3 generations.
Saves up to MAX_IMAGES to: tests/fixtures/images/ai/dalle3/

Usage:
  python3 tests/fixtures/download-dalle3.py
  python3 tests/fixtures/download-dalle3.py --max 400

Requires: datasets, Pillow
  pip install datasets Pillow
"""

import argparse
import random
from pathlib import Path

HF_REPO = "OpenDatasets/dalle-3-dataset"
OUT_DIR = Path(__file__).parent / "images" / "ai" / "dalle3"
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

    ds = load_dataset(HF_REPO, split="train", streaming=True)

    saved = 0
    skipped = 0
    for row in ds:
        if saved >= needed:
            break
        img = row.get("image")
        if img is None:
            skipped += 1
            continue
        w, h = img.size
        if min(w, h) < MIN_SIZE_PX:
            skipped += 1
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = OUT_DIR / f"dalle3_{existing + saved:04d}.jpg"
        img.save(fname, "JPEG", quality=92)
        saved += 1
        if saved % 50 == 0:
            print(f"  Saved {saved}/{needed}...")

    print(f"\n── Done ──")
    print(f"  Saved: {saved} new images")
    print(f"  Skipped (too small or missing): {skipped}")
    print(f"  Total: {existing + saved}")
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
