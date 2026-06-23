#!/usr/bin/env python3
"""
Evaluation script for the L6 forensic NPR model.

Uses held-out test dirs (*-test siblings of each training dir) that are
never seen during training. Produces per-generator and overall metrics.

Outputs lib/forensic-eval.json with:
  - overall:      AUC, precision, recall, FPR, F1 across all test images
  - by_generator: per-source breakdown (recall for AI, FPR for real)
  - threshold_sweep: metrics at [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Usage:
  .venv/bin/python3 tests/fixtures/eval-forensic.py
  dvc repro eval-forensic
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO     = Path(__file__).parent.parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "images"
OUT_PATH = REPO / "lib" / "forensic-eval.json"
EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

# Held-out test dirs (never used in training) → generator label
AI_TEST_DIRS = [
    (FIXTURES / "ai" / "dalle3-test",             "dalle3"),
    (FIXTURES / "ai" / "defactify" / "sd21-test", "stable-diffusion-sd21"),
    (FIXTURES / "ai" / "defactify" / "sd3-test",  "stable-diffusion-sd3"),
    (FIXTURES / "ai" / "defactify" / "sdxl-test", "stable-diffusion-sdxl"),
    (FIXTURES / "ai" / "grok-test",               "grok-aurora"),
    (FIXTURES / "ai" / "midjourney-test",          "midjourney"),
    (FIXTURES / "ai" / "kaggle-test",              "kaggle-ai"),
]

REAL_TEST_DIRS = [
    (FIXTURES / "real" / "coco-test",   "coco"),
    (FIXTURES / "real" / "sun397-test", "sun397"),
    (FIXTURES / "real" / "kaggle-test", "kaggle-real"),
]

# Loose scorecard fixtures — hand-picked, always evaluated separately
SCORECARD_AI = {
    "aurora-astronaut.webp":      "grok-aurora",
    "aurora-cherry-blossom.webp": "grok-aurora",
    "aurora-cyberpunk-city.webp": "grok-aurora",
    "aurora-mountain.webp":       "grok-aurora",
    "aurora-tea-dog.webp":        "grok-aurora",
    "aurora-vangogh-cat.webp":    "grok-aurora",
    "bing-creator-puppy.jpg":     "bing",
    "chatgpt-image.png":          "chatgpt",
    "dalle-blue-man.png":         "dalle3",
    "dalle-puppy.webp":           "dalle3",
    "dalle-robot-letter.png":     "dalle3",
    "eyes.jpg":                   "misc-ai",
    "firefly-tabby-cat.jpg":      "firefly",
    "flux-grid.jpg":              "flux",
    "flux-schnell-grid.jpg":      "flux",
    "sd-android.png":             "stable-diffusion-sdxl",
    "sd-golem.jpg":               "stable-diffusion-sd21",
    "sd-img2img-mountains.png":   "stable-diffusion-sd21",
    "sd-txt2img-01.png":          "stable-diffusion-sd21",
    "sdxl-sample-01.jpg":         "stable-diffusion-sdxl",
    "sdxl-test.png":              "stable-diffusion-sdxl",
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


def apply_npr(arr):
    import numpy as np
    h, w = arr.shape[:2]
    ys = (np.arange(h) & ~1)
    xs = (np.arange(w) & ~1)
    return ((arr - arr[np.ix_(ys, xs)]) * (2.0 / 3.0)).astype("float32")


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
    return apply_npr(np.array(img, dtype="float32") / 255.0)


def load_dir(root, generator, img_size):
    root = Path(root)
    if not root.exists():
        print(f"  MISSING: {root.relative_to(FIXTURES)}", file=sys.stderr)
        return []
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    rows = []
    for p in files:
        try:
            rows.append((p.name, generator, preprocess(p, img_size)))
        except Exception as e:
            print(f"  WARN {p.name}: {e}", file=sys.stderr)
    return rows


def score_batch(model, rows, label, batch_size=32):
    import numpy as np
    results = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        arrays = np.stack([r[2] for r in batch])
        scores = model.predict(arrays, verbose=0, batch_size=batch_size).flatten()
        for (name, gen, _), score in zip(batch, scores):
            results.append((name, gen, label, float(score)))
    return results


def metrics_at(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for l, s in zip(labels, scores):
        p = s >= threshold
        if l == 1 and p:  tp += 1
        elif l == 0 and p: fp += 1
        elif l == 0:       tn += 1
        else:              fn += 1
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(threshold=threshold,
                recall=round(recall, 4), precision=round(precision, 4),
                fpr=round(fpr, 4), f1=round(f1, 4),
                tp=tp, fp=fp, tn=tn, fn=fn)


def auc(labels, scores):
    import numpy as np
    thresholds = sorted(set(scores), reverse=True)
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
        tprs.append(tp / n_pos if n_pos else 0)
        fprs.append(fp / n_neg if n_neg else 0)
    tprs.append(1.0); fprs.append(1.0)
    return float(np.trapz(tprs, fprs))


def print_table(rows, threshold):
    all_labels = [r[2] for r in rows]
    all_scores = [r[3] for r in rows]
    m = metrics_at(all_labels, all_scores, threshold)
    a = auc(all_labels, all_scores)
    print(f"  AUC={a:.4f}  Recall={m['recall']:.3f}  Precision={m['precision']:.3f}"
          f"  FPR={m['fpr']:.3f}  F1={m['f1']:.3f}"
          f"  (n_ai={sum(all_labels)}, n_real={len(all_labels)-sum(all_labels)})")

    generators = sorted(set(r[1] for r in rows))
    ai_gens   = [(g, [r for r in rows if r[1] == g]) for g in generators if rows[[r[1] for r in rows].index(g)][2] == 1]
    real_gens = [(g, [r for r in rows if r[1] == g]) for g in generators if rows[[r[1] for r in rows].index(g)][2] == 0]

    # Precision and FPR can't be computed per-generator in isolation —
    # they require FP/TN counts which only exist across the full real set.
    print(f"\n  {'Generator':<35} {'N':>5}  {'Recall':>7}  {'TP':>4}  {'FN':>4}")
    print("  " + "-" * 57)
    for gen, grp in sorted(ai_gens, key=lambda x: -metrics_at([r[2] for r in x[1]], [r[3] for r in x[1]], threshold)["recall"]):
        m2 = metrics_at([r[2] for r in grp], [r[3] for r in grp], threshold)
        print(f"  {gen:<35} {len(grp):>5}  {m2['recall']:>7.3f}  {m2['tp']:>4}  {m2['fn']:>4}")

    print(f"\n  {'Real source':<35} {'N':>5}  {'FPR':>7}  {'TNR':>7}  {'FP':>4}  {'TN':>4}")
    print("  " + "-" * 62)
    for gen, grp in sorted(real_gens, key=lambda x: metrics_at([r[2] for r in x[1]], [r[3] for r in x[1]], threshold)["fpr"]):
        m2 = metrics_at([r[2] for r in grp], [r[3] for r in grp], threshold)
        print(f"  {gen:<35} {len(grp):>5}  {m2['fpr']:>7.3f}  {1-m2['fpr']:>7.3f}  {m2['fp']:>4}  {m2['tn']:>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size",  type=int,   default=224)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       type=str,   default=str(OUT_PATH))
    args = parser.parse_args()

    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image  # noqa
    except ImportError as e:
        print(f"Missing dependency: {e}"); raise SystemExit(1)

    model_path = REPO / "lib" / "forensic-ai-detector.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr); raise SystemExit(1)

    print(f"Loading {model_path.name}…")
    model = tf.keras.models.load_model(str(model_path))

    # ── Load test dirs ─────────────────────────────────────────────────────────
    print("\n── Test dirs ──")
    dir_rows = []
    for src, gen in AI_TEST_DIRS:
        rows = load_dir(src, gen, args.img_size)
        scored = score_batch(model, rows, label=1)
        dir_rows.extend(scored)
        m = metrics_at([r[2] for r in scored], [r[3] for r in scored], args.threshold)
        print(f"  {str(src.relative_to(FIXTURES)):<40} n={len(scored):>4}  recall={m['recall']:.3f}")

    for src, gen in REAL_TEST_DIRS:
        rows = load_dir(src, gen, args.img_size)
        scored = score_batch(model, rows, label=0)
        dir_rows.extend(scored)
        m = metrics_at([r[2] for r in scored], [r[3] for r in scored], args.threshold)
        print(f"  {str(src.relative_to(FIXTURES)):<40} n={len(scored):>4}  fpr={m['fpr']:.3f}")

    # ── Load scorecard fixtures ────────────────────────────────────────────────
    print("\n── Scorecard fixtures ──")
    scorecard_rows = []
    for fname, gen in SCORECARD_AI.items():
        p = FIXTURES / "ai" / fname
        if not p.exists(): print(f"  MISSING: {fname}", file=sys.stderr); continue
        try:
            x = preprocess(p, args.img_size)[np.newaxis]
            score = float(model.predict(x, verbose=0)[0][0])
            scorecard_rows.append((fname, gen, 1, score))
        except Exception as e:
            print(f"  WARN {fname}: {e}", file=sys.stderr)

    for fname, gen in SCORECARD_REAL.items():
        p = FIXTURES / "real" / fname
        if not p.exists(): print(f"  MISSING: {fname}", file=sys.stderr); continue
        try:
            x = preprocess(p, args.img_size)[np.newaxis]
            score = float(model.predict(x, verbose=0)[0][0])
            scorecard_rows.append((fname, gen, 0, score))
        except Exception as e:
            print(f"  WARN {fname}: {e}", file=sys.stderr)

    all_rows = dir_rows + scorecard_rows

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n╔══ Overall (test dirs + scorecard) ════════════════════╗")
    print_table(all_rows, args.threshold)

    print("\n╔══ Test dirs only ══════════════════════════════════════╗")
    print_table(dir_rows, args.threshold)

    print("\n╔══ Scorecard fixtures only ════════════════════════════╗")
    print_table(scorecard_rows, args.threshold)

    print("\n╔══ Threshold sweep (overall) ══════════════════════════╗")
    all_labels = [r[2] for r in all_rows]
    all_scores = [r[3] for r in all_rows]
    print(f"  {'Threshold':>10}  {'Recall':>7}  {'Precision':>10}  {'FPR':>6}  {'F1':>6}")
    print("  " + "-" * 48)
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = metrics_at(all_labels, all_scores, t)
        mark = " ◄" if t == args.threshold else ""
        print(f"  {t:>10.1f}  {m['recall']:>7.3f}  {m['precision']:>10.3f}"
              f"  {m['fpr']:>6.3f}  {m['f1']:>6.3f}{mark}")

    # ── Build per-generator breakdown for JSON ─────────────────────────────────
    def gen_breakdown(rows):
        out = {}
        for gen in sorted(set(r[1] for r in rows)):
            grp = [r for r in rows if r[1] == gen]
            labels = [r[2] for r in grp]; scores = [r[3] for r in grp]
            is_ai = labels[0] == 1
            m = metrics_at(labels, scores, args.threshold)
            out[gen] = {
                "type": "ai" if is_ai else "real", "n": len(grp),
                "recall":    m["recall"]    if is_ai else None,
                "precision": m["precision"] if is_ai else None,
                "fpr":       m["fpr"]       if not is_ai else None,
                "f1":        m["f1"]        if is_ai else None,
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            }
        return out

    def summary(rows):
        labels = [r[2] for r in rows]; scores = [r[3] for r in rows]
        m = metrics_at(labels, scores, args.threshold)
        return {
            "auc":       round(auc(labels, scores), 4),
            "recall":    m["recall"], "precision": m["precision"],
            "fpr":       m["fpr"],    "f1":        m["f1"],
            "threshold": args.threshold,
            "n_ai":   sum(labels), "n_real": len(labels) - sum(labels),
        }

    all_labels = [r[2] for r in all_rows]
    all_scores = [r[3] for r in all_rows]
    sweep = [metrics_at(all_labels, all_scores, t)
             for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

    result = {
        "overall":    summary(all_rows),
        "test_dirs":  summary(dir_rows),
        "scorecard":  summary(scorecard_rows),
        "by_generator": gen_breakdown(all_rows),
        "threshold_sweep": sweep,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
