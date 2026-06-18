#!/usr/bin/env python3
"""
Download images from DiffusionDB (CC0 license) for training data.
Downloads zip shards directly from HuggingFace without using the datasets library.

Each shard contains ~1000 Stable Diffusion generated images in webp format.
Saves to: tests/fixtures/images/ai/diffusiondb/

Usage:
  python3 tests/fixtures/download-diffusiondb.py             # 1 shard (~1000 images, 527MB)
  python3 tests/fixtures/download-diffusiondb.py --shards 2  # 2 shards (~2000 images)
"""

import argparse
import urllib.request
import zipfile
import shutil
import sys
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"
SHARD_TEMPLATE = "diffusiondb-large-part-1/part-{n:06d}.zip"
SHARD_SIZE_MB = 527

TMP_DIR = Path("/tmp/diffusiondb-dl")
OUT_DIR = Path(__file__).parent / "images" / "ai" / "diffusiondb"


def download_shard(shard_num: int, dest: Path) -> bool:
    url = f"{HF_BASE}/{SHARD_TEMPLATE.format(n=shard_num)}"
    print(f"Downloading shard {shard_num:06d} (~{SHARD_SIZE_MB}MB)...")
    print(f"  {url}")

    def progress(count, block, total):
        pct = min(100, count * block * 100 // max(total, 1))
        print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def extract_shard(zip_path: Path, out_dir: Path, offset: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(('.webp', '.png', '.jpg', '.jpeg'))]
        for name in names:
            dst = out_dir / f"diffusiondb-{offset + extracted:04d}.webp"
            if not dst.exists():
                data = zf.read(name)
                dst.write_bytes(data)
            extracted += 1
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to download (each ~1000 images)')
    args = parser.parse_args()

    TMP_DIR.mkdir(exist_ok=True)
    existing = list(OUT_DIR.glob("*.webp"))
    print(f"Existing DiffusionDB images: {len(existing)}")

    total = 0
    for i in range(1, args.shards + 1):
        zip_path = TMP_DIR / f"part-{i:06d}.zip"

        if not zip_path.exists():
            ok = download_shard(i, zip_path)
            if not ok:
                print(f"Failed to download shard {i}, stopping.")
                break
        else:
            print(f"Using cached shard {i}: {zip_path}")

        n = extract_shard(zip_path, OUT_DIR, offset=len(list(OUT_DIR.glob("*.webp"))))
        total += n
        print(f"  Extracted {n} images from shard {i}")

    print(f"\nDone: {total} new images → {OUT_DIR}")
    print(f"Total in directory: {len(list(OUT_DIR.glob('*.webp')))}")


if __name__ == "__main__":
    main()
