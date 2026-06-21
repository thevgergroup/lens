#!/usr/bin/env python3
"""
Download real (non-AI) photographs for training balance.

Sources:
  - detection-datasets/coco (val split): COCO everyday scenes, ~480–640px
  - tanganke/sun397 (test split): indoor/outdoor scene photos, 768–1024px

Saves to:
  tests/fixtures/images/real/coco/
  tests/fixtures/images/real/sun397/

Usage:
  python3 tests/fixtures/download-real-photos.py
  python3 tests/fixtures/download-real-photos.py --max-per-source 1000

Requires: datasets, Pillow
  pip install datasets Pillow
"""

import argparse
import random
from pathlib import Path

SOURCES = {
    "coco":   ("detection-datasets/coco",   "val",  320),  # COCO images can be 480px
    "sun397": ("tanganke/sun397",            "test", 512),  # SUN397 skews larger
}
OUT_BASE = Path(__file__).parent / "images" / "real"


def download_source(name, repo, split, min_size, max_images):
    from datasets import load_dataset

    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    needed = max_images - existing
    if needed <= 0:
        print(f"  {name}: already have {existing}, skipping")
        return existing

    print(f"  Streaming {repo} ({split})...")
    print(f"  Need {needed} more (have {existing}, target {max_images})")

    ds = load_dataset(repo, split=split, streaming=True)

    saved = 0
    skipped = 0
    for row in ds:
        if saved >= needed:
            break
        img = row.get("image") or row.get("img") or row.get("pixel_values")
        if img is None:
            skipped += 1
            continue
        if not hasattr(img, 'size'):
            skipped += 1
            continue
        w, h = img.size
        if min(w, h) < min_size:
            skipped += 1
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = out_dir / f"{name}_{existing + saved:04d}.jpg"
        img.save(fname, "JPEG", quality=92)
        saved += 1
        if saved % 100 == 0:
            print(f"    Saved {saved}/{needed}...")

    total = existing + saved
    print(f"  {name}: +{saved} new, {skipped} skipped ({total} total)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        from datasets import load_dataset  # noqa
        from PIL import Image  # noqa
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install datasets Pillow")
        raise SystemExit(1)

    print(f"Downloading real photos (target: {args.max_per_source} per source)\n")
    totals = {}
    for name, (repo, split, min_size) in SOURCES.items():
        totals[name] = download_source(name, repo, split, min_size, args.max_per_source)

    print(f"\n── Final counts ──")
    grand = 0
    for name, count in totals.items():
        print(f"  {name}: {count}")
        grand += count
    print(f"  Total: {grand}")


if __name__ == "__main__":
    main()
