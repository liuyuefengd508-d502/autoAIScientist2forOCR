"""
Train the seed TinyCRNN on REAL CASIA-HWDB2 handwritten line images
(Teklia/CASIA-HWDB2-line, validation split, 8318 lines).

For initial benchmarking we split the 8318 validation lines into 7000 train
+ 1318 val. The full train.parquet (876MB) can be added later.
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ----------------------------- device -----------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")


# ----------------------------- data -----------------------------
PARQUET_PATH = glob.glob(
    "./hf_cache/datasets--Teklia--CASIA-HWDB2-line/snapshots/*/data/validation.parquet"
)
assert PARQUET_PATH, "validation parquet not found; run hf_hub_download first"
PARQUET_PATH = PARQUET_PATH[0]
print(f"[data] parquet: {PARQUET_PATH}")


def load_rows():
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(PARQUET_PATH)
    rows = []
    for batch in pf.iter_batches(batch_size=512, columns=["text", "image"]):
        df = batch.to_pandas()
        for _, r in df.iterrows():
            img_bytes = r["image"]["bytes"]
            text = r["text"]
            rows.append((img_bytes, text))
    return rows


print("[data] loading parquet ...")
t0 = time.time()
rows = load_rows()
print(f"[data] {len(rows)} lines loaded in {time.time()-t0:.1f}s")

# Build vocab from data
all_chars = sorted({c for _, t in rows for c in t})
print(f"[vocab] {len(all_chars)} unique chars")
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(all_chars)}
IDX2CHAR = {i + 1: c for i, c in enumerate(all_chars)}
NUM_CLASSES = len(all_chars) + 1


# Train/val split: 7000 / 1318 (deterministic by seed=0)
random.seed(0)
random.shuffle(rows)
TRAIN_ROWS = rows[:7000]
VAL_ROWS = rows[7000:]
print(f"[split] train={len(TRAIN_ROWS)} val={len(VAL_ROWS)}")


def img_bytes_to_tensor(img_bytes: bytes, target_h: int = 48) -> torch.Tensor:
    """Decode + grayscale + resize to height=target_h preserving aspect."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    if img.height != target_h:
        new_w = max(8, int(img.width * target_h / img.height))
        img = img.resize((new_w, target_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)


class CasiaLines(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_bytes, text = self.rows[idx]
        x = img_bytes_to_tensor(img_bytes, target_h=48)
        ids = [CHAR2IDX[c] for c in text if c in CHAR2IDX]
        y = torch.tensor(ids, dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    H = xs[0].shape[1]
    max_w = max(x.shape[-1] for x in xs)
    pad = torch.full((len(xs), 1, H, max_w), -1.0)
    for i, x in enumerate(xs):
        pad[i, :, :, : x.shape[-1]] = x
    targets = torch.cat(ys)
    target_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return pad, targets, target_lens


# ----------------------------- model -----------------------------
class TinyCRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=192):
        super().__init__()
        c = 32
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c * 2, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c * 2, c * 4, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c * 4, c * 4, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c * 4, c * 8, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c * 8, c * 8, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c * 8, c * 8, (3, 1), 1, 0), nn.GELU(),
        )
        self.rnn = nn.LSTM(c * 8, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = f.squeeze(2).transpose(1, 2)
        f, _ = self.rnn(f)
        return self.head(f)


# ----------------------------- decode / metrics -----------------------------
def greedy_decode(logits):
    pred = logits.argmax(-1).detach().cpu().numpy()
    out = []
    for row in pred:
        s = []
        prev = -1
        for v in row:
            if v != prev and v != BLANK:
                s.append(IDX2CHAR.get(int(v), ""))
            prev = v
        out.append("".join(s))
    return out


def edit_distance(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def compute_cer(preds, gts):
    edits = sum(edit_distance(p, g) for p, g in zip(preds, gts))
    total = sum(len(g) for g in gts) or 1
    return edits / total


# ----------------------------- training -----------------------------
@dataclass
class Config:
    seed: int = 0
    batch_size: int = 16
    epochs: int = 6
    lr: float = 5e-4
    grad_clip: float = 5.0


def main(cfg=None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    print(f"[cfg] {cfg}")

    train_ds = CasiaLines(TRAIN_ROWS)
    val_ds = CasiaLines(VAL_ROWS)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = TinyCRNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M num_classes={NUM_CLASSES}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        losses, n = [], 0
        for x, tgt, tgt_len in train_loader:
            x = x.to(DEVICE)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            T_len = torch.full((x.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs.cpu(), tgt, T_len, tgt_len)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            losses.append(float(loss.detach().cpu()))
            n += 1
            if n % 50 == 0:
                print(f"  [ep {ep} step {n}/{len(train_loader)}] loss={np.mean(losses[-50:]):.4f}")
        print(f"[epoch {ep}] train_loss={np.mean(losses):.4f}  ({time.time()-t0:.0f}s elapsed)")

    train_time_min = (time.time() - t0) / 60.0

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, tgt, tgt_len in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(greedy_decode(logits))
            offset = 0
            for L in tgt_len.tolist():
                gts.append("".join(IDX2CHAR.get(int(t), "") for t in tgt[offset : offset + L]))
                offset += L
    cer = compute_cer(preds, gts)

    try:
        peak_mem_gb = torch.mps.driver_allocated_memory() / (1024 ** 3)
    except Exception:
        peak_mem_gb = float("nan")

    metrics = {
        "cer": round(cer, 4),
        "params_M": round(n_params / 1e6, 3),
        "num_classes": NUM_CLASSES,
        "train_time_min": round(train_time_min, 3),
        "peak_mem_gb": round(peak_mem_gb, 3) if peak_mem_gb == peak_mem_gb else None,
        "device": str(DEVICE),
        "n_train": len(TRAIN_ROWS),
        "n_val": len(VAL_ROWS),
        "epochs": cfg.epochs,
        "dataset": "Teklia/CASIA-HWDB2-line (validation split, 7000/1318)",
    }
    print("\nFINAL_METRICS=" + json.dumps(metrics, ensure_ascii=False))

    print("\nSample predictions:")
    for i in range(min(8, len(preds))):
        ok = "✓" if preds[i] == gts[i] else "✗"
        print(f"  {ok} pred={preds[i][:30]!r} gt={gts[i][:30]!r}")

    return metrics


if __name__ == "__main__":
    main()
