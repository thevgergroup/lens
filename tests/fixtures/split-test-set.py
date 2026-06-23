#!/usr/bin/env python3
"""
One-time script: carve out a deterministic 20% test split from each large
image source directory into a sibling *-test directory.

Split is deterministic: files sorted by name, last 20% go to test.
Small dirs (< MIN_SIZE) are left untouched — too few to split.

Run once, then `dvc add` the new test dirs and remove them from train deps.

Usage:
  .venv/bin/python3 tests/fixtures/split-test-set.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

REPO     = Path(__file__).parent.parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "images"
EXTS     = {".jpg", ".jpeg", ".png", ".webp"}
MIN_SIZE = 30   # dirs smaller than this are skipped
TEST_FRAC = 0.20

SOURCES = [
    FIXTURES / "ai"   / "dalle3",
    FIXTURES / "ai"   / "defactify" / "sd21",
    FIXTURES / "ai"   / "defactify" / "sd3",
    FIXTURES / "ai"   / "defactify" / "sdxl",
    FIXTURES / "ai"   / "grok",
    FIXTURES / "ai"   / "midjourney",
    FIXTURES / "ai"   / "kaggle",
    FIXTURES / "real" / "coco",
    FIXTURES / "real" / "sun397",
    FIXTURES / "real" / "kaggle",
]


def split_dir(src: Path, dry_run: bool):
    files = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    )
    if len(files) < MIN_SIZE:
        print(f"  SKIP {src.relative_to(FIXTURES)}  ({len(files)} files < {MIN_SIZE})")
        return

    n_test = max(1, int(len(files) * TEST_FRAC))
    test_files = files[-n_test:]   # last N by sorted name → deterministic

    # Sibling dir: dalle3 → dalle3-test, defactify/sd21 → defactify/sd21-test
    dest = src.parent / (src.name + "-test")

    print(f"  {str(src.relative_to(FIXTURES)):<35} {len(files):>5} total  →  {n_test} to {dest.relative_to(FIXTURES)}")

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for f in test_files:
            shutil.move(str(f), dest / f.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be moved\n")

    print(f"{'Source':<40} {'Total':>7}  Split")
    print("-" * 60)
    for src in SOURCES:
        if not src.exists():
            print(f"  MISSING: {src.relative_to(FIXTURES)}")
            continue
        split_dir(src, args.dry_run)

    if args.dry_run:
        print("\nRe-run without --dry-run to apply.")
    else:
        print("\nDone. Next steps:")
        print("  dvc add tests/fixtures/images/ai/dalle3-test")
        print("  dvc add tests/fixtures/images/ai/defactify/sd21-test")
        print("  dvc add tests/fixtures/images/ai/defactify/sd3-test")
        print("  dvc add tests/fixtures/images/ai/defactify/sdxl-test")
        print("  dvc add tests/fixtures/images/ai/grok-test")
        print("  dvc add tests/fixtures/images/ai/midjourney-test")
        print("  dvc add tests/fixtures/images/ai/kaggle-test")
        print("  dvc add tests/fixtures/images/real/coco-test")
        print("  dvc add tests/fixtures/images/real/sun397-test")
        print("  dvc add tests/fixtures/images/real/kaggle-test")
        print("  dvc push")


if __name__ == "__main__":
    main()
