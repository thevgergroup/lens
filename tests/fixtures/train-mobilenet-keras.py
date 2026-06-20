#!/usr/bin/env python3
"""
Train MobileNetV3-Small in Keras/TF and export directly to TF.js.

Two-stage fine-tuning:
  Stage 1: Head only (frozen backbone), 12 epochs
  Stage 2: Last 15 layers unfrozen, 12 epochs (early stop, patience=4)

Output:
  lib/mobilenet-tfjs/model.json + group1-shard1of1.bin  (~2.3 MB total)
  lib/mobilenet-ai-detector-keras.keras  (checkpoint)

Usage:
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-mobilenet-keras.py
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-mobilenet-keras.py --dry-run
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-mobilenet-keras.py --no-wandb
"""

import argparse
import os
import sys
import random
import warnings
import unittest.mock

warnings.filterwarnings('ignore')
sys.modules['tensorflow_decision_forests'] = unittest.mock.MagicMock()

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_auc_score

FIXTURES = Path(__file__).parent / 'images'
OUT_DIR  = Path(__file__).parent.parent.parent / 'lib'

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def collect_images(root, label, exclude_prefix=None, exclude_dirs=None):
    exclude_dirs = set(exclude_dirs or [])
    paths = []
    for p in Path(root).rglob('*'):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        if exclude_prefix and p.name.startswith(exclude_prefix):
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        paths.append((str(p), label))
    return paths


def load_image(path, augment=False):
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
        w, h = img.size
        left = (w - IMG_SIZE) // 2
        top  = (h - IMG_SIZE) // 2
        img = img.crop((left, top, left+IMG_SIZE, top+IMG_SIZE))

        x = np.array(img, dtype=np.float32) / 255.0

        if augment:
            if random.random() > 0.5:
                x = x[:, ::-1, :]
            if random.random() > 0.5:
                x = np.clip(x * random.uniform(0.9, 1.1), 0, 1)

        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


def make_dataset(samples, augment=False, batch_size=32, shuffle=True):
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    def gen():
        indices = list(range(len(paths)))
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            yield load_image(paths[i], augment=augment), labels[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=512)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )
    x = tf.keras.layers.Dense(256, activation='hard_sigmoid')(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(base.input, x), base


def evaluate_auc(model, val_ds, val_samples):
    probs, labels = [], []
    for batch_x, batch_y in val_ds:
        p = model(batch_x, training=False).numpy().flatten()
        probs.extend(p)
        labels.extend(batch_y.numpy())
    return roc_auc_score(labels, probs), np.array(probs), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-head', type=int, default=12)
    parser.add_argument('--epochs-ft',   type=int, default=12)
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--dry-run',       action='store_true')
    parser.add_argument('--no-wandb',      action='store_true', help='Disable wandb logging')
    parser.add_argument('--run-name',      type=str, default=None, help='wandb run name')
    parser.add_argument('--stage1-only',   action='store_true', help='Skip Stage 2 fine-tuning')
    parser.add_argument('--exclude-ai-dirs', nargs='+', default=[], help='AI subdirs to exclude')
    args = parser.parse_args()

    if args.dry_run:
        args.epochs_head = 2
        args.epochs_ft   = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Exclude cashbowman from both sides — that Kaggle scrape mixed genuine AI
    # images with blog thumbnails, screenshots of AI tools, and editorial photos,
    # poisoning both classes. Only use hand-curated images.
    ai_samples   = collect_images(FIXTURES / 'ai',   1, exclude_prefix='cashbowman_',
                                  exclude_dirs=set(args.exclude_ai_dirs))
    # Exclude coco/sun397 — domain too similar to Defactify AI images (both COCO-style scenes),
    # confuses the classifier. Curated hand-labeled real images only.
    real_samples = collect_images(FIXTURES / 'real', 0, exclude_prefix='cashbowman_',
                                  exclude_dirs={'coco', 'sun397'})
    print(f'AI: {len(ai_samples)}  Real: {len(real_samples)}')

    # Count by source directory for wandb config
    def count_by_source(samples, base):
        counts = {}
        for path, _ in samples:
            rel = Path(path).relative_to(base)
            src = rel.parts[0] if len(rel.parts) > 1 else 'root'
            counts[src] = counts.get(src, 0) + 1
        return counts

    ai_by_source   = count_by_source(ai_samples,   FIXTURES / 'ai')
    real_by_source = count_by_source(real_samples, FIXTURES / 'real')

    # 80/20 stratified split
    random.shuffle(ai_samples)
    random.shuffle(real_samples)
    ai_train,   ai_val   = ai_samples[:int(.8*len(ai_samples))],   ai_samples[int(.8*len(ai_samples)):]
    real_train, real_val = real_samples[:int(.8*len(real_samples))], real_samples[int(.8*len(real_samples)):]

    train_samples = ai_train + real_train
    val_samples   = ai_val   + real_val
    random.shuffle(train_samples)

    n_ai   = sum(l for _,l in train_samples)
    n_real = len(train_samples) - n_ai
    print(f'Train: {len(train_samples)} ({n_ai} AI)  Val: {len(val_samples)}')

    # Class weights to handle imbalance
    class_weight = {0: 1.0, 1: n_real / n_ai}

    # ── wandb init ────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project='lens-ai-detector',
                name=args.run_name,
                config={
                    'model': 'MobileNetV3Small',
                    'img_size': IMG_SIZE,
                    'epochs_head': args.epochs_head,
                    'epochs_ft': args.epochs_ft,
                    'batch_size': args.batch,
                    'lr_head': 1e-3,
                    'lr_ft': 2e-5,
                    'ft_layers': 0 if args.stage1_only else 15,
                    'stage1_only': args.stage1_only,
                    'early_stop_patience': 4,
                    'exclude_ai_dirs': args.exclude_ai_dirs,
                    'seed': args.seed,
                    'n_ai_total': len(ai_samples),
                    'n_real_total': len(real_samples),
                    'n_train': len(train_samples),
                    'n_val': len(val_samples),
                    'class_weight_ai': n_real / n_ai,
                    **{f'ai_{k}': v for k, v in ai_by_source.items()},
                    **{f'real_{k}': v for k, v in real_by_source.items()},
                }
            )
        except Exception as e:
            print(f'wandb init failed: {e} — continuing without logging')
            use_wandb = False

    train_ds = make_dataset(train_samples, augment=True,  batch_size=args.batch, shuffle=True)
    val_ds   = make_dataset(val_samples,   augment=False, batch_size=args.batch, shuffle=False)

    model, base = build_model()

    # ── Stage 1: head only ────────────────────────────────────────────────────
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    print(f'\n── Stage 1: head only ({args.epochs_head} epochs) ──')
    best_auc, best_weights = 0, None
    for epoch in range(1, args.epochs_head + 1):
        model.fit(train_ds, epochs=1, verbose=0, class_weight=class_weight)
        auc, _, _ = evaluate_auc(model, val_ds, val_samples)
        marker = ' *' if auc > best_auc else ''
        print(f'  epoch {epoch:2d}/{args.epochs_head}  val_auc={auc:.3f}{marker}')
        if use_wandb:
            wandb.log({'stage': 1, 'epoch': epoch, 'val_auc': auc})
        if auc > best_auc:
            best_auc = auc
            best_weights = model.get_weights()

    model.set_weights(best_weights)
    print(f'  Best AUC: {best_auc:.3f}')
    if use_wandb:
        wandb.log({'stage1_best_auc': best_auc})

    # ── Stage 2: unfreeze last 15 layers ─────────────────────────────────────
    best_auc2, best_weights2 = best_auc, best_weights
    if args.stage1_only:
        print('\n── Skipping Stage 2 (--stage1-only) ──')
    else:
        base.trainable = True
        for layer in base.layers[:-15]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-5),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        print(f'\n── Stage 2: fine-tune last 15 layers ({args.epochs_ft} epochs) ──')
        patience, no_improve = 4, 0
        for epoch in range(1, args.epochs_ft + 1):
            model.fit(train_ds, epochs=1, verbose=0, class_weight=class_weight)
            auc, _, _ = evaluate_auc(model, val_ds, val_samples)
            marker = ''
            if auc > best_auc2:
                best_auc2 = auc
                best_weights2 = model.get_weights()
                no_improve = 0
                marker = ' *'
            else:
                no_improve += 1
            print(f'  epoch {epoch:2d}/{args.epochs_ft}  val_auc={auc:.3f}{marker}')
            if use_wandb:
                wandb.log({'stage': 2, 'epoch': args.epochs_head + epoch, 'val_auc': auc})
            if no_improve >= patience:
                print(f'  Early stop (no improvement for {patience} epochs)')
                break

    model.set_weights(best_weights2)

    # ── Final evaluation ──────────────────────────────────────────────────────
    auc, probs, labels = evaluate_auc(model, val_ds, val_samples)
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0

    print(f'\n── Final ──')
    print(f'AUC={auc:.4f}  Precision={prec:.3f}  Recall={rec:.3f}')
    print(f'Threshold sweep:')

    threshold_data = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p = (probs>=t).astype(int)
        tp_ = int(((p==1)&(labels==1)).sum())
        fp_ = int(((p==1)&(labels==0)).sum())
        fn_ = int(((p==0)&(labels==1)).sum())
        tn_ = int(((p==0)&(labels==0)).sum())
        tpr  = tp_/(tp_+fn_) if tp_+fn_ else 0
        fpr  = fp_/(fp_+tn_) if fp_+tn_ else 0
        prec_ = tp_/(tp_+fp_) if tp_+fp_ else 0
        print(f'  t={t:.1f}  TPR={tpr:.3f}  FPR={fpr:.3f}  Prec={prec_:.3f}')
        threshold_data.append({'threshold': t, 'tpr': tpr, 'fpr': fpr, 'precision': prec_})

    if use_wandb:
        wandb.log({
            'final_auc': auc,
            'final_precision': prec,
            'final_recall': rec,
            'stage2_best_auc': best_auc2,
            'threshold_table': wandb.Table(
                columns=['threshold', 'tpr', 'fpr', 'precision'],
                data=[[d['threshold'], d['tpr'], d['fpr'], d['precision']] for d in threshold_data]
            ),
        })

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = OUT_DIR / 'mobilenet-ai-detector-keras.keras'
    model.save(str(ckpt_path))
    print(f'\nCheckpoint saved → {ckpt_path}')

    # ── Export to TF.js ───────────────────────────────────────────────────────
    tfjs_dir = OUT_DIR / 'mobilenet-tfjs'
    tfjs_dir.mkdir(exist_ok=True)

    model.export(str(OUT_DIR / 'mobilenet-keras-export'))
    tfjs.converters.convert_tf_saved_model(
        str(OUT_DIR / 'mobilenet-keras-export'),
        str(tfjs_dir),
        quantization_dtype_map={'float16': '*'},
    )
    files = list(tfjs_dir.iterdir())
    total = sum(f.stat().st_size for f in files)
    print(f'TF.js model → {tfjs_dir}  ({total/1e6:.2f} MB)')
    for f in files:
        print(f'  {f.name}: {f.stat().st_size/1e6:.3f} MB')

    if use_wandb:
        wandb.log({'model_size_mb': total / 1e6})
        wandb.finish()


if __name__ == '__main__':
    main()
