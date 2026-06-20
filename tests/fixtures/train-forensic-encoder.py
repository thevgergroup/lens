#!/usr/bin/env python3
"""
Train a forensic-signal CNN for AI image detection.

Architecture: NPR transform → DepthwiseSeparable CNN (~400K params, ~800KB float16)

Key difference from MobileNetV3 approach:
  - Input is the NPR residual (content-suppressed upsampling artifact map)
    rather than ImageNet-normalized RGB. The model learns forensic signals,
    not scene semantics.
  - Trained from scratch — ImageNet pretraining is counterproductive here
    because it biases the encoder toward semantic features.
  - JPEG augmentation at q=75-95 to handle CDN re-encoding (LinkedIn, Twitter)

Reference: "Rethinking the Up-Sampling Operations in CNN-based Generative
Network for Generalizable Deepfake Detection" (NPR, CVPR 2024)

Output:
  lib/forensic-tfjs/model.json + *.bin  (~800KB float16)
  lib/forensic-ai-detector.keras

Usage:
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-forensic-encoder.py
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-forensic-encoder.py --dry-run
  /tmp/tfjs-venv/bin/python3 tests/fixtures/train-forensic-encoder.py --no-wandb
"""

import argparse
import io
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
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

FIXTURES = Path(__file__).parent / 'images'
OUT_DIR  = Path(__file__).parent.parent.parent / 'lib'

IMG_SIZE = 224
EXCLUDE_REAL_DIRS = {'coco', 'sun397'}


# ── NPR Transform ─────────────────────────────────────────────────────────────
# Computes the Neighboring Pixel Residual: downsample 2x nearest-neighbor,
# upsample 2x, subtract from original. Isolates upsampling artifacts that
# all generative models (GANs, diffusion) leave behind. Content is suppressed.

def apply_npr(img_array):
    """img_array: float32 [H, W, 3] in [0,1]. Returns NPR residual same shape."""
    h, w = img_array.shape[:2]
    # Nearest-neighbor 2x downsample then 2x upsample
    # Equivalent to: each pixel takes the value of its top-left 2x2 neighbor
    # Broadcasting trick: floor x,y to even → creates the reconstruction
    ys = (np.arange(h) & ~1)  # floor to even
    xs = (np.arange(w) & ~1)
    reconstructed = img_array[np.ix_(ys, xs)]  # [H, W, 3]
    residual = (img_array - reconstructed) * (2.0 / 3.0)
    return residual.astype(np.float32)


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


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def load_image(path, augment=False):
    try:
        img = Image.open(path).convert('RGB')

        # JPEG compression augmentation — critical for CDN robustness
        # Real-world images on LinkedIn/Twitter are re-encoded at q=75-85
        if augment and random.random() > 0.4:
            q = random.randint(75, 95)
            img = jpeg_compress(img, q)

        # Resize keeping aspect ratio then center crop to IMG_SIZE
        w, h = img.size
        scale = (IMG_SIZE + 32) / min(w, h)  # resize so short side = IMG_SIZE+32
        nw, nh = max(IMG_SIZE, int(w * scale)), max(IMG_SIZE, int(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        w, h = img.size

        if augment:
            # Random crop instead of center crop
            left = random.randint(0, w - IMG_SIZE)
            top  = random.randint(0, h - IMG_SIZE)
        else:
            left = (w - IMG_SIZE) // 2
            top  = (h - IMG_SIZE) // 2
        img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

        x = np.array(img, dtype=np.float32) / 255.0

        if augment:
            if random.random() > 0.5:
                x = x[:, ::-1, :]  # horizontal flip
            # Mild brightness/contrast — keep subtle so NPR signal isn't destroyed
            if random.random() > 0.6:
                x = np.clip(x * random.uniform(0.92, 1.08), 0, 1)

        # Apply NPR transform — this is the key difference from MobileNetV3
        x = apply_npr(x)
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


# ── Model Architecture ────────────────────────────────────────────────────────

def depthwise_sep_block(x, filters, stride=1):
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_forensic_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inp = tf.keras.Input(shape=input_shape)

    # Initial conv — learns what to amplify in the NPR residual
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Depthwise-separable blocks — efficient spatial artifact extraction
    x = depthwise_sep_block(x, 64,  stride=2)
    x = depthwise_sep_block(x, 128, stride=2)
    x = depthwise_sep_block(x, 128, stride=2)
    x = depthwise_sep_block(x, 256, stride=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inp, out)


def evaluate_auc(model, val_ds):
    probs, labels = [], []
    for batch_x, batch_y in val_ds:
        p = model(batch_x, training=False).numpy().flatten()
        probs.extend(p)
        labels.extend(batch_y.numpy())
    return roc_auc_score(labels, probs), np.array(probs), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-head', type=int, default=15, help='Training epochs (single stage)')
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--dry-run',     action='store_true')
    parser.add_argument('--no-wandb',    action='store_true')
    parser.add_argument('--run-name',    type=str, default='forensic-npr')
    parser.add_argument('--exclude-ai-dirs', nargs='+', default=[])
    args = parser.parse_args()

    if args.dry_run:
        args.epochs_head = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ai_samples   = collect_images(FIXTURES / 'ai',   1,
                                  exclude_prefix='cashbowman_',
                                  exclude_dirs=set(args.exclude_ai_dirs))
    real_samples = collect_images(FIXTURES / 'real', 0,
                                  exclude_prefix='cashbowman_',
                                  exclude_dirs=EXCLUDE_REAL_DIRS)
    print(f'AI: {len(ai_samples)}  Real: {len(real_samples)}')

    def count_by_source(samples, base):
        counts = {}
        for path, _ in samples:
            rel = Path(path).relative_to(base)
            src = rel.parts[0] if len(rel.parts) > 1 else 'root'
            counts[src] = counts.get(src, 0) + 1
        return counts

    ai_by_source   = count_by_source(ai_samples,   FIXTURES / 'ai')
    real_by_source = count_by_source(real_samples, FIXTURES / 'real')
    print('AI sources:',   ai_by_source)
    print('Real sources:', real_by_source)

    random.shuffle(ai_samples)
    random.shuffle(real_samples)

    ai_train,   ai_val   = ai_samples[:int(.8*len(ai_samples))],   ai_samples[int(.8*len(ai_samples)):]
    real_train, real_val = real_samples[:int(.8*len(real_samples))], real_samples[int(.8*len(real_samples)):]

    train_samples = ai_train + real_train
    val_samples   = ai_val   + real_val
    random.shuffle(train_samples)

    n_ai   = sum(l for _, l in train_samples)
    n_real = len(train_samples) - n_ai
    print(f'Train: {len(train_samples)} ({n_ai} AI, {n_real} real)  Val: {len(val_samples)}')

    class_weight = {0: 1.0, 1: n_real / n_ai}
    print(f'Class weight AI: {class_weight[1]:.3f}')

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project='lens-ai-detector',
                name=args.run_name,
                config={
                    'model': 'ForensicNPR-DepthwiseSep',
                    'preprocessing': 'NPR residual (2/3 scale)',
                    'img_size': IMG_SIZE,
                    'epochs': args.epochs_head,
                    'batch_size': args.batch,
                    'lr': args.lr,
                    'jpeg_aug': True,
                    'jpeg_quality_range': '75-95',
                    'seed': args.seed,
                    'n_ai': len(ai_samples),
                    'n_real': len(real_samples),
                    'class_weight_ai': class_weight[1],
                    **{f'ai_{k}': v for k, v in ai_by_source.items()},
                    **{f'real_{k}': v for k, v in real_by_source.items()},
                }
            )
        except Exception as e:
            print(f'wandb init failed: {e} — continuing without logging')
            use_wandb = False

    train_ds = make_dataset(train_samples, augment=True,  batch_size=args.batch, shuffle=True)
    val_ds   = make_dataset(val_samples,   augment=False, batch_size=args.batch, shuffle=False)

    model = build_forensic_model()
    model.summary(print_fn=lambda s: print(f'  {s}'))

    total_params = model.count_params()
    print(f'\nTotal params: {total_params:,}  (~{total_params*2/1e6:.2f}MB float16)')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    print(f'\n── Training ({args.epochs_head} epochs, from scratch) ──')
    best_auc, best_weights = 0, None
    patience, no_improve = 5, 0

    for epoch in range(1, args.epochs_head + 1):
        model.fit(train_ds, epochs=1, verbose=0, class_weight=class_weight)
        auc, _, _ = evaluate_auc(model, val_ds)
        marker = ''
        if auc > best_auc:
            best_auc = auc
            best_weights = model.get_weights()
            no_improve = 0
            marker = ' *'
        else:
            no_improve += 1
        print(f'  epoch {epoch:2d}/{args.epochs_head}  val_auc={auc:.4f}{marker}')
        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'val_auc': auc})
        if no_improve >= patience:
            print(f'  Early stop (no improvement for {patience} epochs)')
            break

    model.set_weights(best_weights)

    # ── Final evaluation ──────────────────────────────────────────────────────
    auc, probs, labels = evaluate_auc(model, val_ds)
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
        import wandb
        wandb.log({
            'final_auc': auc,
            'final_precision': prec,
            'final_recall': rec,
            'best_auc': best_auc,
            'threshold_table': wandb.Table(
                columns=['threshold', 'tpr', 'fpr', 'precision'],
                data=[[d['threshold'], d['tpr'], d['fpr'], d['precision']] for d in threshold_data]
            ),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt_path = OUT_DIR / 'forensic-ai-detector.keras'
    model.save(str(ckpt_path))
    print(f'\nCheckpoint → {ckpt_path}')

    tfjs_dir = OUT_DIR / 'forensic-tfjs'
    tfjs_dir.mkdir(exist_ok=True)
    model.export(str(OUT_DIR / 'forensic-keras-export'))
    tfjs.converters.convert_tf_saved_model(
        str(OUT_DIR / 'forensic-keras-export'),
        str(tfjs_dir),
        quantization_dtype_map={'float16': '*'},
    )
    files = list(tfjs_dir.iterdir())
    total = sum(f.stat().st_size for f in files)
    print(f'TF.js model → {tfjs_dir}  ({total/1e6:.2f} MB)')
    for f in sorted(files):
        print(f'  {f.name}: {f.stat().st_size/1e6:.3f} MB')

    if use_wandb:
        import wandb
        wandb.log({'model_size_mb': total / 1e6})
        wandb.finish()


if __name__ == '__main__':
    main()
