#!/usr/bin/env python3
"""
Download SynthBuster dataset from Zenodo.
- 9,000 AI images (1,000 per generator × 9 generators)
- 1,000 paired real photos (RAISE-1k)

Generators: dalle2, dalle3, firefly, midjourney, sd1.3, sd1.4, sd2, sdxl, glide
Saves to:
  tests/fixtures/images/ai/synthbuster/<generator>/
  tests/fixtures/images/real/synthbuster/
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
OUT_DIR    = Path(__file__).parent / "images"
TMP_ZIP    = Path("/tmp/synthbuster.zip")
TMP_UNZIP  = Path("/tmp/synthbuster-unzip")

# Generators we want — all of them
AI_GENERATORS = ["dalle2", "dalle3", "firefly", "midjourney", "sd1.3", "sd1.4", "sd2", "sdxl", "glide"]
MAX_PER_GEN   = 500  # cap to keep training set manageable; 9 × 500 = 4500 AI images

def download(url, dest):
    print(f"Downloading {url}")
    print(f"  → {dest}")
    def progress(count, block, total):
        pct = min(100, count * block * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()

def main():
    if not TMP_ZIP.exists():
        download(ZENODO_URL, TMP_ZIP)
    else:
        print(f"Using cached zip: {TMP_ZIP}")

    print(f"\nExtracting to {TMP_UNZIP}...")
    if TMP_UNZIP.exists():
        shutil.rmtree(TMP_UNZIP)
    TMP_UNZIP.mkdir()
    with zipfile.ZipFile(TMP_ZIP) as zf:
        zf.extractall(TMP_UNZIP)

    # Discover structure
    extracted = list(TMP_UNZIP.rglob("*"))
    dirs = [p for p in extracted if p.is_dir()]
    print(f"Extracted dirs: {[str(d.relative_to(TMP_UNZIP)) for d in dirs[:20]]}")

    # Find AI image directories
    n_ai, n_real = 0, 0

    for gen in AI_GENERATORS:
        # Try common directory patterns
        candidates = (
            list(TMP_UNZIP.rglob(f"*/{gen}")) +
            list(TMP_UNZIP.rglob(f"*{gen}*"))
        )
        src_dirs = [d for d in candidates if d.is_dir()]

        if not src_dirs:
            print(f"  WARNING: no directory found for generator '{gen}'")
            continue

        src_dir = src_dirs[0]
        dst_dir = OUT_DIR / "ai" / "synthbuster" / gen
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in src_dir.rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        images = images[:MAX_PER_GEN]

        for img in images:
            shutil.copy2(img, dst_dir / img.name)
            n_ai += 1

        print(f"  {gen}: {len(images)} images → {dst_dir.relative_to(Path.cwd())}")

    # Real images (RAISE-1k)
    real_candidates = (
        list(TMP_UNZIP.rglob("*/real")) +
        list(TMP_UNZIP.rglob("*raise*")) +
        list(TMP_UNZIP.rglob("*RAISE*"))
    )
    real_dirs = [d for d in real_candidates if d.is_dir()]

    if real_dirs:
        dst_real = OUT_DIR / "real" / "synthbuster"
        dst_real.mkdir(parents=True, exist_ok=True)
        real_images = sorted(p for p in real_dirs[0].rglob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for img in real_images[:1000]:
            shutil.copy2(img, dst_real / img.name)
            n_real += 1
        print(f"  real (RAISE-1k): {n_real} images → {dst_real.relative_to(Path.cwd())}")
    else:
        # Dump all files so we can inspect
        print("  WARNING: could not find real image directory. Top-level contents:")
        for p in sorted(TMP_UNZIP.iterdir()):
            print(f"    {p.name}/")

    print(f"\nDone: {n_ai} AI images, {n_real} real images")

if __name__ == "__main__":
    main()
