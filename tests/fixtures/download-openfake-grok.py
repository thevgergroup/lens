#!/usr/bin/env python3
"""
Download Grok-2 (Aurora) images from ComplexDataLab/OpenFake (HuggingFace).

Opens one Parquet shard at a time over HTTP via fsspec, applies pyarrow
predicate pushdown on the 'model' column so image bytes for non-grok rows
are never decoded. Stops when --max total images exist in the output directory.

License: CC-BY-NC-4.0 (non-commercial use only)
Paper:   https://arxiv.org/abs/2509.09495

Usage:
  python3 tests/fixtures/download-openfake-grok.py
  python3 tests/fixtures/download-openfake-grok.py --max 300
"""

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

TARGET  = "grok-2-image-1212"
HF_API  = "https://huggingface.co/api/datasets/ComplexDataLab/OpenFake"
HF_BASE = "https://huggingface.co/datasets/ComplexDataLab/OpenFake/resolve/main"
OUT_DIR = Path(__file__).parent / "images" / "ai" / "grok"


def get_train_shards():
    req = urllib.request.Request(HF_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        meta = json.load(r)
    return [
        f"{HF_BASE}/{s['rfilename']}"
        for s in meta["siblings"]
        if s["rfilename"].startswith("core/train") and s["rfilename"].endswith(".parquet")
    ]


def iter_grok_images(shard_url):
    """Yield PIL images for grok-2-image-1212 rows.

    Two-pass approach:
      1. Read only the 'model' column (tiny) to find which row groups contain
         grok rows — this is cheap even over HTTP.
      2. Read image+model for only those row groups, skipping all others.
    """
    import fsspec
    import pyarrow.parquet as pq
    from PIL import Image

    with fsspec.open(shard_url, "rb", timeout=120) as f:
        pf = pq.ParquetFile(f)

        # Pass 1: scan model column across all row groups to find grok row groups.
        # model column is 0 KB compressed so this is cheap even over HTTP.
        grok_row_groups = []
        for rg in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg, columns=["model"])
            models = batch["model"].to_pylist()
            if TARGET in models:
                grok_row_groups.append(rg)

        # Pass 2: read image+model only for row groups that contain grok rows
        for rg in grok_row_groups:
            batch = pf.read_row_group(rg, columns=["image", "model"])
            for i in range(batch.num_rows):
                if batch["model"][i].as_py() != TARGET:
                    continue
                img_struct = batch["image"][i].as_py()
                if img_struct is None:
                    continue
                raw = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct
                if raw:
                    yield Image.open(io.BytesIO(raw)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=300, help="Target total images in output dir")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.jpg")) + list(OUT_DIR.glob("*.png")) + list(OUT_DIR.glob("*.webp")))
    needed = args.max - existing
    if needed <= 0:
        print(f"Already have {existing} images, nothing to do.")
        return

    print(f"Target: {args.max} images (have {existing}, need {needed} more)")
    print("Fetching shard list...")
    shards = get_train_shards()
    print(f"Found {len(shards)} train shards")

    saved = 0
    next_idx = existing

    for shard_idx, shard_url in enumerate(shards):
        if saved >= needed:
            break

        print(f"Shard {shard_idx}/{len(shards)-1}...", end=" ", flush=True)
        try:
            shard_saved = 0
            for img in iter_grok_images(shard_url):
                fname = OUT_DIR / f"grok-openfake-{next_idx:04d}.jpg"
                img.save(fname, "JPEG", quality=92)
                next_idx += 1
                saved += 1
                shard_saved += 1
                if saved >= needed:
                    break
            print(f"+{shard_saved}")
        except Exception as e:
            print(f"error: {e} — skipping")
            time.sleep(5)
            continue

        time.sleep(1)  # brief pause between shards to avoid rate-limiting

    print(f"\n── Done ──")
    print(f"  Downloaded: {saved}")
    print(f"  Total grok: {existing + saved}")
    print(f"  Output:     {OUT_DIR}")


if __name__ == "__main__":
    main()
