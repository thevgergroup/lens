#!/usr/bin/env python3
"""
JPEG augmentation stage for the forensic training pipeline.

Reads all AI images from the source directories, saves each one at a
deterministically-seeded JPEG quality factor so the model learns to detect
AI generation artifacts that survive lossy compression.

Real images are intentionally NOT augmented — they are already camera-JPEG or
CDN-JPEG compressed, so they represent the real-world distribution as-is.

Reproducibility: quality factor is seeded from hash(relative_path) so the same
source image always gets the same quality factor, across machines and reruns.
Changing augment.jpeg_quality_min / augment.jpeg_quality_max in params.yaml
busts the DVC cache and forces a re-augment + retrain.

Output: tests/fixtures/images/ai/augmented/
  One JPEG per source image, named <source_dir>__<original_stem>.jpg

Usage:
  .venv/bin/python3 tests/fixtures/augment-forensic.py
  dvc repro augment-forensic
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_DIR   = REPO_ROOT / "tests" / "fixtures" / "images" / "ai" / "augmented"

AI_SOURCE_DIRS = [
    "tests/fixtures/images/ai/defactify",
    "tests/fixtures/images/ai/dalle3",
    "tests/fixtures/images/ai/midjourney",
    "tests/fixtures/images/ai/grok",
    "tests/fixtures/images/ai/kaggle",
    "tests/fixtures/images/ai/training",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def quality_for_path(rel_path: str, q_min: int, q_max: int) -> int:
    """Deterministic quality factor in [q_min, q_max] seeded from file path."""
    h = int(hashlib.md5(rel_path.encode()).hexdigest(), 16)
    return q_min + (h % (q_max - q_min + 1))


def augment_image(src: Path, dest: Path, quality: int) -> bool:
    try:
        from PIL import Image
        img = Image.open(src).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        dest.write_bytes(buf.getvalue())
        return True
    except Exception as e:
        print(f"  WARN: {src.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-min",  type=int, default=40)
    parser.add_argument("--jpeg-max",  type=int, default=95)
    parser.add_argument("--seed",      type=int, default=42,
                        help="Seed mixed into path hash for reproducibility")
    args = parser.parse_args()

    try:
        from PIL import Image  # noqa
    except ImportError:
        print("Missing dependency: pip install Pillow")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = skipped = errors = 0

    for src_dir_rel in AI_SOURCE_DIRS:
        src_dir = REPO_ROOT / src_dir_rel
        if not src_dir.exists():
            print(f"  SKIP (missing): {src_dir_rel}")
            continue

        images = sorted(p for p in src_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            continue

        print(f"{src_dir_rel}  ({len(images)} images)")
        dir_tag = src_dir.name

        for src in images:
            rel_to_src = src.relative_to(src_dir)
            rel = f"{args.seed}/{src_dir_rel}/{rel_to_src}"
            quality = quality_for_path(rel, args.jpeg_min, args.jpeg_max)
            # Flatten subdir structure into filename: dir__subdir__stem.jpg
            parts = [dir_tag] + list(rel_to_src.parent.parts) + [src.stem]
            dest = OUT_DIR / ("__".join(p for p in parts if p != '.') + ".jpg")

            if dest.exists():
                skipped += 1
                continue

            if augment_image(src, dest, quality):
                saved += 1
            else:
                errors += 1

    total = saved + skipped
    print(f"\n── Done ──")
    print(f"  Written:  {saved}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors:   {errors}")
    print(f"  Total:    {total}")
    print(f"  Output:   {OUT_DIR}")


if __name__ == "__main__":
    main()
