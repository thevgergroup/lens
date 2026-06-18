#!/usr/bin/env python3
"""
Download a sample from tristanzhang32/ai-generated-images-vs-real-images (Kaggle).
- AI: SD, Midjourney, DALL-E (sample up to MAX_PER_GEN each)
- Real: Pexels/Unsplash photos (sample up to MAX_REAL)

Saves to:
  tests/fixtures/images/ai/tristanzhang/<generator>/
  tests/fixtures/images/real/tristanzhang/
"""

import os
import sys
import zipfile
import shutil
import random
from pathlib import Path

DATASET   = "tristanzhang32/ai-generated-images-vs-real-images"
TMP_DIR   = Path("/tmp/tristanzhang-dl")
OUT_DIR   = Path(__file__).parent / "images"

MAX_PER_GEN = 400   # AI images per generator (SD / MJ / DALL-E)
MAX_REAL    = 800   # real photos to add

def kaggle_download():
    import subprocess
    TMP_DIR.mkdir(exist_ok=True)
    print(f"Downloading {DATASET} via Kaggle API...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(TMP_DIR), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("Kaggle download failed — check credentials in ~/.kaggle/kaggle.json")
        sys.exit(1)

def main():
    if not any(TMP_DIR.glob("**/*.jpg")) and not any(TMP_DIR.glob("**/*.png")):
        kaggle_download()
    else:
        print(f"Using cached download in {TMP_DIR}")

    # Discover structure
    print("\nTop-level directories:")
    for p in sorted(TMP_DIR.iterdir()):
        n = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 0
        print(f"  {p.name}/  ({n} files)")

    n_ai, n_real = 0, 0

    # --- AI images ---
    # Common patterns: fake/, ai/, generated/, or subdirs named by generator
    ai_root_candidates = (
        list(TMP_DIR.rglob("fake")) +
        list(TMP_DIR.rglob("ai")) +
        list(TMP_DIR.rglob("generated")) +
        list(TMP_DIR.rglob("train/fake")) +
        list(TMP_DIR.rglob("test/fake"))
    )
    ai_roots = [d for d in ai_root_candidates if d.is_dir()]

    if ai_roots:
        ai_root = ai_roots[0]
        print(f"\nAI root: {ai_root.relative_to(TMP_DIR)}")
        subdirs = [d for d in ai_root.iterdir() if d.is_dir()]

        if subdirs:
            # Subdirectories per generator
            for subdir in subdirs:
                gen_name = subdir.name.lower().replace(" ", "_")
                imgs = sorted(p for p in subdir.rglob("*")
                              if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
                random.shuffle(imgs)
                sample = imgs[:MAX_PER_GEN]
                dst = OUT_DIR / "ai" / "tristanzhang" / gen_name
                dst.mkdir(parents=True, exist_ok=True)
                for img in sample:
                    shutil.copy2(img, dst / img.name)
                    n_ai += 1
                print(f"  {gen_name}: {len(sample)} / {len(imgs)} images")
        else:
            # Flat directory — treat as single "mixed" AI set
            imgs = sorted(p for p in ai_root.rglob("*")
                          if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
            random.shuffle(imgs)
            sample = imgs[:MAX_PER_GEN * 3]
            dst = OUT_DIR / "ai" / "tristanzhang" / "mixed"
            dst.mkdir(parents=True, exist_ok=True)
            for img in sample:
                shutil.copy2(img, dst / img.name)
                n_ai += 1
            print(f"  mixed: {len(sample)} / {len(imgs)} images")
    else:
        print("WARNING: could not find AI image directory — dumping structure:")
        for p in TMP_DIR.rglob("*"):
            if p.is_dir():
                print(f"  DIR  {p.relative_to(TMP_DIR)}")

    # --- Real images ---
    real_root_candidates = (
        list(TMP_DIR.rglob("real")) +
        list(TMP_DIR.rglob("train/real")) +
        list(TMP_DIR.rglob("test/real"))
    )
    real_roots = [d for d in real_root_candidates if d.is_dir()]

    if real_roots:
        real_root = real_roots[0]
        print(f"\nReal root: {real_root.relative_to(TMP_DIR)}")
        imgs = sorted(p for p in real_root.rglob("*")
                      if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
        random.shuffle(imgs)
        sample = imgs[:MAX_REAL]
        dst = OUT_DIR / "real" / "tristanzhang"
        dst.mkdir(parents=True, exist_ok=True)
        for img in sample:
            shutil.copy2(img, dst / img.name)
            n_real += 1
        print(f"  real: {len(sample)} / {len(imgs)} images")
    else:
        print("WARNING: could not find real image directory")

    print(f"\nDone: {n_ai} AI images, {n_real} real images")

if __name__ == "__main__":
    random.seed(42)
    main()
