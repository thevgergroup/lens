#!/usr/bin/env python3
"""
Downloads 20 images from DiffusionDB (CC0 license) for pixel statistics testing.

Requires:
  pip install datasets Pillow

These images are used to test Layer 3 (statistical pixel analysis) because they
are unmodified Stable Diffusion outputs with full pixel fidelity (no EXIF stripping).
"""

import os
import sys
from pathlib import Path

DEST_DIR = Path(__file__).parent / "images" / "ai"
NUM_IMAGES = 20


def main():
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install datasets Pillow")
        sys.exit(1)

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(DEST_DIR.glob("diffusiondb-*.png"))
    if len(existing) >= NUM_IMAGES:
        print(f"✓ Already have {len(existing)} DiffusionDB images, skipping download.")
        return

    print(f"Downloading {NUM_IMAGES} images from DiffusionDB (large_random_1k subset)...")
    print("This streams a small slice without downloading the full 2M dataset.\n")

    ds = load_dataset(
        "poloclub/diffusiondb",
        "large_random_1k",
        split="train",
        trust_remote_code=True,
    )

    downloaded = 0
    for i, item in enumerate(ds):
        if downloaded >= NUM_IMAGES:
            break
        dest = DEST_DIR / f"diffusiondb-{downloaded:03d}.png"
        if dest.exists():
            downloaded += 1
            continue
        img = item["image"]
        if img is None:
            continue
        # Ensure minimum size for pixel stats testing (Layer 3 needs >100px, Layer 4 needs 64x64)
        if img.width < 64 or img.height < 64:
            continue
        img.save(dest, "PNG")
        print(f"  ✓ {dest.name} ({img.width}x{img.height})")
        downloaded += 1

    print(f"\nDownloaded {downloaded} images to {DEST_DIR}")


if __name__ == "__main__":
    main()
