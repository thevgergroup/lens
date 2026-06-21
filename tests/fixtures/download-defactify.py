#!/usr/bin/env python3
"""
Download photorealistic AI images from Defactify Image Dataset (HuggingFace).
Dataset: Rajarshi-Roy-research/Defactify_Image_Dataset

Images are MS COCO scenes regenerated with 5 generators:
  Label_B=1: SD 2.1 (768×768)
  Label_B=2: SDXL   (1024×1024)
  Label_B=3: SD 3   (1024×1024)
  Label_B=4: DALL-E 3 (270×270 — SKIPPED, too small)
  Label_B=5: Midjourney v6 (436×436 — SKIPPED, too small)

Saves up to MAX_PER_GEN images per generator to:
  tests/fixtures/images/ai/defactify/<generator>/

Usage:
  python3 tests/fixtures/download-defactify.py
  python3 tests/fixtures/download-defactify.py --shards 2  # download more shards
  python3 tests/fixtures/download-defactify.py --max-per-gen 200

Requires: huggingface_hub, pandas, pyarrow, Pillow
  pip install huggingface_hub pandas pyarrow Pillow
"""

import argparse
import io
import random
from pathlib import Path

HF_REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
TMP_DIR = Path("/tmp/defactify-dl")
OUT_DIR = Path(__file__).parent / "images" / "ai" / "defactify"

# Generators by Label_B — skip tiny ones (DALL-E3=270px, MJ6=436px)
GENERATORS = {
    1: ("sd21",       768),
    2: ("sdxl",      1024),
    3: ("sd3",       1024),
    # 4: ("dalle3",   270),   # too small for training
    # 5: ("midjourney6", 436), # too small for training
}

MIN_SIZE_PX = 512  # skip images smaller than this


def download_shard(shard_num: int, split: str = "train") -> Path:
    from huggingface_hub import hf_hub_download
    total_shards = 7 if split == "train" else 8
    filename = f"data/{split}-{shard_num:05d}-of-{total_shards:05d}.parquet"
    dest = TMP_DIR / f"{split}-{shard_num:05d}.parquet"
    TMP_DIR.mkdir(exist_ok=True)
    if dest.exists():
        print(f"Using cached: {dest.name}")
        return dest
    print(f"Downloading {filename} from {HF_REPO}...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(TMP_DIR / "hf-cache"),
        local_dir=str(TMP_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def extract_images(parquet_path: Path, max_per_gen: int) -> dict:
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows from {parquet_path.name}")

    counts = {}
    for label_b, (gen_name, expected_size) in GENERATORS.items():
        out_dir = OUT_DIR / gen_name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("*.jpg")))
        needed = max_per_gen - existing
        if needed <= 0:
            print(f"  {gen_name}: already have {existing}, skipping")
            counts[gen_name] = existing
            continue

        subset = df[df["Label_B"] == label_b]
        random.shuffle(subset_list := subset.to_dict("records"))

        saved = 0
        for row in subset_list:
            if saved >= needed:
                break
            img_bytes = row["Image"].get("bytes") if isinstance(row["Image"], dict) else None
            if not img_bytes:
                continue
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            if min(w, h) < MIN_SIZE_PX:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            fname = out_dir / f"{gen_name}_{existing + saved:04d}.jpg"
            img.save(fname, "JPEG", quality=92)
            saved += 1

        total = existing + saved
        counts[gen_name] = total
        print(f"  {gen_name}: +{saved} new ({total} total)")

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1, help="Number of train shards to download (each ~448MB)")
    parser.add_argument("--max-per-gen", type=int, default=400, help="Max images per generator")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        import pandas  # noqa
        from PIL import Image  # noqa
        from huggingface_hub import hf_hub_download  # noqa
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install huggingface_hub pandas pyarrow Pillow")
        raise SystemExit(1)

    all_counts = {}
    for shard_i in range(args.shards):
        print(f"\n── Shard {shard_i} ──")
        parquet = download_shard(shard_i, split="train")
        counts = extract_images(parquet, args.max_per_gen)
        for k, v in counts.items():
            all_counts[k] = v

    print("\n── Final counts ──")
    total = 0
    for gen_name, count in all_counts.items():
        print(f"  {gen_name}: {count}")
        total += count
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
