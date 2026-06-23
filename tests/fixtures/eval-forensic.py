#!/usr/bin/env python3
"""
Evaluation script for the L6 forensic NPR model.

Runs the model against:
  - Scorecard fixture images (loose files in tests/fixtures/images/ai/ and real/)
  - Training-source directories (dalle3/, defactify/, grok/, etc.) — sampled

Outputs lib/forensic-eval.json with:
  - Overall metrics (AUC, precision, recall, FPR, F1) at threshold sweep
  - Per-generator breakdown (recall per AI source, FPR per real source)

Usage:
  .venv/bin/python3 tests/fixtures/eval-forensic.py
  dvc repro eval-forensic
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO     = Path(__file__).parent.parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "images"
OUT_PATH = REPO / "lib" / "forensic-eval.json"

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Loose scorecard files: label → generator tag
SCORECARD_AI = {
    "aurora-astronaut.webp":    "aurora",
    "aurora-cherry-blossom.webp": "aurora",
    "aurora-cyberpunk-city.webp": "aurora",
    "aurora-mountain.webp":     "aurora",
    "aurora-tea-dog.webp":      "aurora",
    "aurora-vangogh-cat.webp":  "aurora",
    "bing-creator-puppy.jpg":   "bing",
    "chatgpt-image.png":        "chatgpt",
    "dalle-blue-man.png":       "dalle",
    "dalle-puppy.webp":         "dalle",
    "dalle-robot-letter.png":   "dalle",
    "eyes.jpg":                 "misc-ai",
    "firefly-tabby-cat.jpg":    "firefly",
    "flux-grid.jpg":            "flux",
    "flux-schnell-grid.jpg":    "flux",
    "sd-android.png":           "stable-diffusion",
    "sd-golem.jpg":             "stable-diffusion",
    "sd-img2img-mountains.png": "stable-diffusion",
    "sd-txt2img-01.png":        "stable-diffusion",
    "sdxl-sample-01.jpg":       "stable-diffusion",
    "sdxl-test.png":            "stable-diffusion",
}

SCORECARD_REAL = {
    "ant-photo.jpg":               "photo",
    "car-photo.jpg":               "photo",
    "cat-photo.jpg":               "photo",
    "city-nyc.jpg":                "photo",
    "cloudscape.jpg":              "photo",
    "crater-lake-nocreds.jpg":     "photo",
    "crater-lake.jpg":             "photo",
    "dice-photo.png":              "photo",
    "dog-photo.jpg":               "photo",
    "flower-macro.jpg":            "photo",
    "golden-gate-bridge.jpg":      "photo",
    "lena.png":                    "photo",
    "mountain-landscape.jpg":      "photo",
    "nasa-astronaut-portrait.jpg": "nasa",
    "nasa-earth.jpg":              "nasa",
    "nasa-madagascar-coast.jpg":   "nasa",
    "nasa-mars.jpg":               "nasa",
    "nasa-wildlife.jpg":           "nasa",
    "ocean-waves.jpg":             "photo",
    "picsum-landscape-10.jpg":     "picsum",
    "picsum-landscape-110.jpg":    "picsum",
    "picsum-landscape-65.jpg":     "picsum",
    "picsum-portrait-91.jpg":      "picsum",
    "picsum-scene-200.jpg":        "picsum",
    "waterfall.jpg":               "photo",
}

# Training-source directories → generator tag
AI_SOURCE_DIRS = [
    (FIXTURES / "ai" / "dalle3",     "dalle3-dataset"),
    (FIXTURES / "ai" / "defactify" / "sd21",  "stable-diffusion-dataset"),
    (FIXTURES / "ai" / "defactify" / "sd3",   "stable-diffusion-dataset"),
    (FIXTURES / "ai" / "defactify" / "sdxl",  "stable-diffusion-dataset"),
    (FIXTURES / "ai" / "grok",       "aurora-dataset"),
    (FIXTURES / "ai" / "midjourney", "midjourney-dataset"),
    (FIXTURES / "ai" / "kaggle",     "kaggle-ai-dataset"),
]

REAL_SOURCE_DIRS = [
    (FIXTURES / "real" / "coco",    "coco-dataset"),
    (FIXTURES / "real" / "sun397",  "sun397-dataset"),
    (FIXTURES / "real" / "twitter", "twitter-dataset"),
    (FIXTURES / "real" / "kaggle",  "kaggle-real-dataset"),
]


def apply_npr(img_array):
    import numpy as np
    h, w = img_array.shape[:2]
    ys = (np.arange(h) & ~1)
    xs = (np.arange(w) & ~1)
    reconstructed = img_array[np.ix_(ys, xs)]
    residual = (img_array - reconstructed) * (2.0 / 3.0)
    return residual.astype("float32")


def preprocess(path, img_size):
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = (img_size + 32) / min(w, h)
    nw = max(img_size, int(w * scale))
    nh = max(img_size, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    w, h = img.size
    left = (w - img_size) // 2
    top  = (h - img_size) // 2
    img  = img.crop((left, top, left + img_size, top + img_size))
    x = np.array(img, dtype="float32") / 255.0
    return apply_npr(x)


def collect_dir(root, generator, max_per_dir, img_size, seed):
    import numpy as np
    root = Path(root)
    if not root.exists():
        return []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    rng = random.Random(seed)
    if max_per_dir and len(files) > max_per_dir:
        files = rng.sample(files, max_per_dir)
    results = []
    for p in files:
        try:
            x = preprocess(p, img_size)
            results.append((str(p), generator, x))
        except Exception as e:
            print(f"  WARN: {p.name}: {e}", file=sys.stderr)
    return results


def metrics_at_threshold(labels, scores, threshold):
    import numpy as np
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
                recall=round(recall, 4), precision=round(precision, 4),
                fpr=round(fpr, 4), f1=round(f1, 4))


def compute_auc(labels, scores):
    import numpy as np
    # Trapezoidal AUC from threshold sweep
    thresholds = sorted(set(scores), reverse=True) + [0.0]
    tprs, fprs = [0.0], [0.0]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    for t in thresholds:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
        tprs.append(tp / n_pos if n_pos else 0)
        fprs.append(fp / n_neg if n_neg else 0)
    tprs.append(1.0); fprs.append(1.0)
    return float(np.trapz(tprs, fprs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size",      type=int, default=224)
    parser.add_argument("--threshold",     type=float, default=0.50)
    parser.add_argument("--max-per-dir",   type=int, default=200,
                        help="Max images sampled per training-source dir")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--no-train-dirs", action="store_true",
                        help="Only evaluate scorecard fixtures, skip training dirs")
    parser.add_argument("--out",           type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image  # noqa
    except ImportError as e:
        print(f"Missing dependency: {e}")
        raise SystemExit(1)

    model_path = REPO / "lib" / "forensic-ai-detector.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # ── Scorecard fixtures ─────────────────────────────────────────────────────
    print("\n── Scorecard fixtures ──")
    scorecard_rows = []  # (path, generator, label, score)

    for fname, gen in SCORECARD_AI.items():
        p = FIXTURES / "ai" / fname
        if not p.exists():
            print(f"  MISSING: {fname}", file=sys.stderr)
            continue
        try:
            x = preprocess(p, args.img_size)[np.newaxis]
            score = float(model.predict(x, verbose=0)[0][0])
            scorecard_rows.append((fname, gen, 1, score))
        except Exception as e:
            print(f"  WARN: {fname}: {e}", file=sys.stderr)

    for fname, gen in SCORECARD_REAL.items():
        p = FIXTURES / "real" / fname
        if not p.exists():
            print(f"  MISSING: {fname}", file=sys.stderr)
            continue
        try:
            x = preprocess(p, args.img_size)[np.newaxis]
            score = float(model.predict(x, verbose=0)[0][0])
            scorecard_rows.append((fname, gen, 0, score))
        except Exception as e:
            print(f"  WARN: {fname}: {e}", file=sys.stderr)

    # ── Training-source dirs ───────────────────────────────────────────────────
    dir_rows = []
    if not args.no_train_dirs:
        print("\n── Training-source directories ──")
        for src_dir, gen in AI_SOURCE_DIRS:
            items = collect_dir(src_dir, gen, args.max_per_dir, args.img_size, args.seed)
            if not items:
                continue
            paths, gens, arrays = zip(*items)
            batch = np.stack(arrays)
            scores = model.predict(batch, verbose=0, batch_size=32).flatten().tolist()
            for path, g, score in zip(paths, gens, scores):
                dir_rows.append((Path(path).name, g, 1, score))
            print(f"  {src_dir.relative_to(FIXTURES)}  n={len(items)}")

        for src_dir, gen in REAL_SOURCE_DIRS:
            items = collect_dir(src_dir, gen, args.max_per_dir, args.img_size, args.seed)
            if not items:
                continue
            paths, gens, arrays = zip(*items)
            batch = np.stack(arrays)
            scores = model.predict(batch, verbose=0, batch_size=32).flatten().tolist()
            for path, g, score in zip(paths, gens, scores):
                dir_rows.append((Path(path).name, g, 0, score))
            print(f"  {src_dir.relative_to(FIXTURES)}  n={len(items)}")

    all_rows = scorecard_rows + dir_rows

    # ── Per-generator breakdown ────────────────────────────────────────────────
    generators = sorted(set(r[1] for r in all_rows))
    gen_breakdown = {}
    for gen in generators:
        rows = [r for r in all_rows if r[1] == gen]
        labels = [r[2] for r in rows]
        scores = [r[3] for r in rows]
        m = metrics_at_threshold(labels, scores, args.threshold)
        is_ai = labels[0] == 1
        gen_breakdown[gen] = {
            "type":      "ai" if is_ai else "real",
            "n":         len(rows),
            "recall":    m["recall"] if is_ai else None,
            "fpr":       m["fpr"]    if not is_ai else None,
            "precision": m["precision"] if is_ai else None,
            "f1":        m["f1"] if is_ai else None,
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        }

    # ── Overall metrics ────────────────────────────────────────────────────────
    all_labels = [r[2] for r in all_rows]
    all_scores = [r[3] for r in all_rows]

    scorecard_labels = [r[2] for r in scorecard_rows]
    scorecard_scores = [r[3] for r in scorecard_rows]

    sweep = [metrics_at_threshold(all_labels, all_scores, t)
             for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

    overall = metrics_at_threshold(all_labels, all_scores, args.threshold)
    auc     = compute_auc(all_labels, all_scores)

    scorecard_overall = metrics_at_threshold(scorecard_labels, scorecard_scores, args.threshold)
    scorecard_auc     = compute_auc(scorecard_labels, scorecard_scores)

    # ── Print tables ───────────────────────────────────────────────────────────
    print("\n╔══ Overall (all images) ════════════════════════════════╗")
    print(f"  AUC={auc:.4f}  Recall={overall['recall']:.3f}  Precision={overall['precision']:.3f}"
          f"  FPR={overall['fpr']:.3f}  F1={overall['f1']:.3f}")
    print(f"  n_ai={sum(all_labels)}  n_real={len(all_labels)-sum(all_labels)}")

    print("\n╔══ Scorecard only (fixture images) ════════════════════╗")
    print(f"  AUC={scorecard_auc:.4f}  Recall={scorecard_overall['recall']:.3f}"
          f"  Precision={scorecard_overall['precision']:.3f}"
          f"  FPR={scorecard_overall['fpr']:.3f}  F1={scorecard_overall['f1']:.3f}")
    print(f"  n_ai={sum(scorecard_labels)}  n_real={len(scorecard_labels)-sum(scorecard_labels)}")

    print("\n╔══ Per-generator breakdown ════════════════════════════╗")
    ai_gens   = [(g, v) for g, v in gen_breakdown.items() if v["type"] == "ai"]
    real_gens = [(g, v) for g, v in gen_breakdown.items() if v["type"] == "real"]

    print(f"\n  {'Generator':<30} {'N':>5}  {'Recall':>7}  {'Precision':>10}  {'F1':>6}")
    print("  " + "-" * 60)
    for gen, v in sorted(ai_gens, key=lambda x: -(x[1]["recall"] or 0)):
        print(f"  {gen:<30} {v['n']:>5}  {v['recall']:>7.3f}  {v['precision']:>10.3f}  {v['f1']:>6.3f}")

    print(f"\n  {'Source':<30} {'N':>5}  {'FPR':>7}  {'TNR':>7}")
    print("  " + "-" * 40)
    for gen, v in sorted(real_gens, key=lambda x: (x[1]["fpr"] or 0)):
        tnr = 1.0 - (v["fpr"] or 0)
        print(f"  {gen:<30} {v['n']:>5}  {v['fpr']:>7.3f}  {tnr:>7.3f}")

    print("\n╔══ Threshold sweep (overall) ══════════════════════════╗")
    print(f"  {'Threshold':>10}  {'Recall':>7}  {'Precision':>10}  {'FPR':>6}  {'F1':>6}")
    print("  " + "-" * 48)
    for m in sweep:
        mark = " ◄" if m["threshold"] == args.threshold else ""
        print(f"  {m['threshold']:>10.1f}  {m['recall']:>7.3f}  {m['precision']:>10.3f}"
              f"  {m['fpr']:>6.3f}  {m['f1']:>6.3f}{mark}")

    # ── Write JSON ─────────────────────────────────────────────────────────────
    out = {
        "overall": {
            "auc":       round(auc, 4),
            "recall":    overall["recall"],
            "precision": overall["precision"],
            "fpr":       overall["fpr"],
            "f1":        overall["f1"],
            "threshold": args.threshold,
            "n_ai":      sum(all_labels),
            "n_real":    len(all_labels) - sum(all_labels),
        },
        "scorecard": {
            "auc":       round(scorecard_auc, 4),
            "recall":    scorecard_overall["recall"],
            "precision": scorecard_overall["precision"],
            "fpr":       scorecard_overall["fpr"],
            "f1":        scorecard_overall["f1"],
            "n_ai":      sum(scorecard_labels),
            "n_real":    len(scorecard_labels) - sum(scorecard_labels),
        },
        "by_generator": gen_breakdown,
        "threshold_sweep": sweep,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
