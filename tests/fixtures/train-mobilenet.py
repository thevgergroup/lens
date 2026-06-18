#!/usr/bin/env python3
"""
Fine-tune MobileNetV3-Small to classify AI-generated vs real images.

Stage 1: Train classifier head only (frozen backbone), 10 epochs
Stage 2: Unfreeze last 2 conv blocks, fine-tune at lower LR, 10 epochs

Usage:
  python3 tests/fixtures/train-mobilenet.py
  python3 tests/fixtures/train-mobilenet.py --epochs-head 15 --epochs-ft 15
  python3 tests/fixtures/train-mobilenet.py --dry-run   # quick smoke test
"""

import argparse
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, UnidentifiedImageError
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

FIXTURES = Path(__file__).parent / 'images'
OUT_DIR  = Path(__file__).parent.parent.parent / 'lib'

# ImageNet stats — required since we start from ImageNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── dataset ──────────────────────────────────────────────────────────────────

def collect_images(root, label, exclude_prefix=None):
    paths = []
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            if exclude_prefix and p.name.startswith(exclude_prefix):
                continue
            paths.append((str(p), label))
    return paths


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


def make_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Mild colour jitter — don't destroy noise fingerprints
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ── model ─────────────────────────────────────────────────────────────────────

def build_model():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace classifier head: 576 → 256 → 1
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    return model


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model, n=3):
    # Unfreeze last n feature blocks
    blocks = list(model.features.children())
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True


# ── training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += ((logits > 0) == (labels > 0.5)).sum().item()
        n          += len(imgs)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    auc    = roc_auc_score(labels, probs)
    acc    = (preds == labels).mean()
    return auc, acc, probs, labels


def run_stage(name, model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([val_loader.dataset.samples.count((s,1)) /
                                  max(1, val_loader.dataset.samples.count((s,0)))
                                  for s in []]).to(device)
        if False else torch.ones(1).to(device)
    )
    print(f'\n── {name} ──')
    best_auc, best_state = 0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        marker = ' *' if val_auc > best_auc else ''
        print(f'  epoch {epoch:2d}/{epochs}  loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  '
              f'val_auc={val_auc:.3f}  val_acc={val_acc:.3f}{marker}')
        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_auc


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-head', type=int, default=12)
    parser.add_argument('--epochs-ft',   type=int, default=12)
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--dry-run',     action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        args.epochs_head = 2
        args.epochs_ft   = 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f'Device: {device}')

    # Collect images — exclude cashbowman stock photos from real set
    ai_samples   = collect_images(FIXTURES / 'ai',   1)
    real_samples = collect_images(FIXTURES / 'real', 0, exclude_prefix='cashbowman_')
    print(f'AI: {len(ai_samples)}  Real: {len(real_samples)}')

    all_samples = ai_samples + real_samples
    random.shuffle(all_samples)

    # 80/20 stratified split
    ai_train   = ai_samples[:int(0.8*len(ai_samples))]
    ai_val     = ai_samples[int(0.8*len(ai_samples)):]
    real_train = real_samples[:int(0.8*len(real_samples))]
    real_val   = real_samples[int(0.8*len(real_samples)):]
    train_samples = ai_train + real_train
    val_samples   = ai_val   + real_val
    random.shuffle(train_samples)

    print(f'Train: {len(train_samples)} ({sum(l for _,l in train_samples)} AI)  '
          f'Val: {len(val_samples)} ({sum(l for _,l in val_samples)} AI)')

    # Weighted sampler to handle class imbalance
    n_ai   = sum(l for _, l in train_samples)
    n_real = len(train_samples) - n_ai
    weights = [1/n_ai if l == 1 else 1/n_real for _, l in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)

    train_ds = ImageDataset(train_samples, make_transforms(augment=True))
    val_ds   = ImageDataset(val_samples,   make_transforms(augment=False))
    # Attach samples list for reference
    val_ds.samples = val_samples

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model().to(device)

    # ── Stage 1: head only ────────────────────────────────────────────────────
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_head)
    auc1 = run_stage('Stage 1: head only', model, train_loader, val_loader,
                     optimizer, scheduler, args.epochs_head, device)

    # ── Stage 2: unfreeze last 3 blocks ──────────────────────────────────────
    unfreeze_last_blocks(model, n=3)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_ft)
    auc2 = run_stage('Stage 2: fine-tune last 3 blocks', model, train_loader, val_loader,
                     optimizer, scheduler, args.epochs_ft, device)

    # ── Final evaluation ─────────────────────────────────────────────────────
    print('\n── Final evaluation on validation set ──')
    auc, acc, probs, labels = evaluate(model, val_loader, device)
    print(f'AUC={auc:.4f}  Acc={acc:.4f}')
    preds = (probs >= 0.5).astype(int)
    print(classification_report(labels, preds, target_names=['Real', 'AI']))

    # Threshold sweep
    print('Threshold sweep (FPR / TPR):')
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = (probs >= t).astype(int)
        tp = ((p==1) & (labels==1)).sum()
        fp = ((p==1) & (labels==0)).sum()
        fn = ((p==0) & (labels==1)).sum()
        tn = ((p==0) & (labels==0)).sum()
        print(f'  t={t:.1f}  TPR={tp/(tp+fn):.3f}  FPR={fp/(fp+tn):.3f}  '
              f'Prec={tp/(tp+fp) if tp+fp>0 else 0:.3f}')

    # Save model weights
    out_path = OUT_DIR / 'mobilenet-ai-detector.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'auc': auc,
        'acc': acc,
        'n_train': len(train_samples),
        'n_val':   len(val_samples),
    }, out_path)
    print(f'\nModel saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)')

    # Also export to ONNX for browser deployment
    onnx_path = OUT_DIR / 'mobilenet-ai-detector.onnx'
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    try:
        # torch >= 2.12 prefers export_for_inference / dynamo path
        ep = torch.onnx.export(
            model, (dummy,), onnx_path,
            input_names=['image'], output_names=['logit'],
            dynamic_shapes={'image': {0: torch.export.Dim('batch')}},
        )
        print(f'ONNX exported → {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)')
    except Exception as e:
        # Fallback to legacy API
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=['image'], output_names=['logit'],
            dynamic_axes={'image': {0: 'batch'}, 'logit': {0: 'batch'}},
            opset_version=17,
            do_constant_folding=True,
        )
        print(f'ONNX exported (legacy) → {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
