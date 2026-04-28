"""
Train TinyCRNN on CASIA-HWDB2 validation lines, but filtered to the
top-N most-frequent characters only.

Motivation: full vocab = 2630 classes, only 7 K lines → 2.7 samples/class →
CTC collapses. Keeping top-100 chars still covers ~80% of Chinese text
and gives ~60 samples/class after filtering.
"""
from __future__ import annotations

import collections
import glob
import io
import json
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")

PARQUET_PATH = glob.glob(
    "./hf_cache/datasets--Teklia--CASIA-HWDB2-line/snapshots/*/data/validation.parquet"
)[0]


def load_rows():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_PATH)
    rows = []
    for batch in pf.iter_batches(batch_size=512, columns=["text", "image"]):
        df = batch.to_pandas()
        for _, r in df.iterrows():
            rows.append((r["image"]["bytes"], r["text"]))
    return rows


print("[data] loading parquet …")
t0 = time.time()
all_rows = load_rows()
print(f"[data] {len(all_rows)} lines in {time.time()-t0:.1f}s")

# ---- build freq table ----
freq = collections.Counter(c for _, t in all_rows for c in t)
print(f"[vocab] full size={len(freq)}")

MIN_FREQ = 20   # keep chars appearing >= MIN_FREQ times; rest → <UNK>
UNK_CH = "\x00"  # sentinel for unknown chars
keep_chars = sorted(c for c, n in freq.items() if n >= MIN_FREQ)
VOCAB = set(keep_chars)
top_chars = keep_chars + [UNK_CH]  # UNK is the last class
coverage = sum(v for c, v in freq.items() if c in VOCAB) / sum(freq.values())
print(f"[vocab] freq>={MIN_FREQ}: {len(keep_chars)} chars, covers {coverage*100:.1f}% of all char tokens")

# ---- keep all rows; map out-of-vocab chars to UNK ----
def normalize(t):
    return "".join(c if c in VOCAB else UNK_CH for c in t)

clean_rows = [(b, normalize(t)) for b, t in all_rows if len(t) > 0]
print(f"[data] {len(clean_rows)} lines (all kept, rare chars→UNK)")

BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(top_chars)}
IDX2CHAR = {i + 1: c for i, c in enumerate(top_chars)}
NUM_CLASSES = len(top_chars) + 1

random.seed(0)
random.shuffle(clean_rows)
n_train = int(len(clean_rows) * 0.85)
TRAIN_ROWS = clean_rows[:n_train]
VAL_ROWS = clean_rows[n_train:]
print(f"[split] train={len(TRAIN_ROWS)} val={len(VAL_ROWS)} classes={NUM_CLASSES}")


def img_to_tensor(img_bytes: bytes, h: int = 48) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    if img.height != h:
        w = max(8, int(img.width * h / img.height))
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5).unsqueeze(0)


class CasiaLines(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        b, t = self.rows[idx]
        x = img_to_tensor(b, 48)
        y = torch.tensor([CHAR2IDX[c] for c in t], dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    H = xs[0].shape[1]
    max_w = max(x.shape[-1] for x in xs)
    pad = torch.full((len(xs), 1, H, max_w), -1.0)
    for i, x in enumerate(xs):
        pad[i, :, :, :x.shape[-1]] = x
    return pad, torch.cat(ys), torch.tensor([len(y) for y in ys], dtype=torch.long)


class TinyCRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden=256):
        super().__init__()
        c = 32
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c*2, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c*2, c*4, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c*4, c*4, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c*4, c*8, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c*8, c*8, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c*8, c*8, (3, 1), 1, 0), nn.GELU(),
        )
        self.rnn = nn.LSTM(c*8, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        f, _ = self.rnn(f)
        return self.head(f)


def greedy_decode(logits):
    pred = logits.argmax(-1).detach().cpu().numpy()
    result = []
    for row in pred:
        s, prev = [], -1
        for v in row:
            if v != prev and v != BLANK:
                s.append(IDX2CHAR.get(int(v), ""))
            prev = v
        result.append("".join(s))
    return result


def cer(preds, gts):
    def ed(a, b):
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        dp = list(range(len(b) + 1))
        for ca in a:
            prev, dp[0] = dp[0], dp[0] + 1
            for j, cb in enumerate(b, 1):
                cur = dp[j]
                dp[j] = min(dp[j]+1, dp[j-1]+1, prev+(ca != cb))
                prev = cur
        return dp[-1]
    return sum(ed(p, g) for p, g in zip(preds, gts)) / max(1, sum(len(g) for g in gts))


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 16
    epochs: int = 12
    lr: float = 3e-4
    grad_clip: float = 5.0


def main(cfg: Config | None = None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    print(f"[cfg] {cfg}")

    train_dl = DataLoader(CasiaLines(TRAIN_ROWS), cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(CasiaLines(VAL_ROWS),   cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = TinyCRNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M classes={NUM_CLASSES}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, tgt, tgt_len in train_dl:
            x = x.to(DEVICE)
            lp = F.log_softmax(model(x), dim=-1).transpose(0, 1)
            T  = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
            loss = ctc(lp.cpu(), tgt, T, tgt_len)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        print(f"[epoch {ep:2d}] loss={np.mean(losses):.4f}  lr={scheduler.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    train_min = (time.time()-t0)/60

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, tgt, tgt_len in val_dl:
            preds.extend(greedy_decode(model(x.to(DEVICE))))
            off = 0
            for L in tgt_len.tolist():
                gts.append("".join(IDX2CHAR.get(int(c),"") for c in tgt[off:off+L])); off+=L

    val_cer = cer(preds, gts)
    try:
        mem = torch.mps.driver_allocated_memory() / 1024**3
    except Exception:
        mem = float("nan")

    m = {
        "cer": round(val_cer, 4),
        "params_M": round(n_params/1e6, 3),
        "num_classes": NUM_CLASSES,
        "n_train": len(TRAIN_ROWS),
        "n_val": len(VAL_ROWS),
        "train_time_min": round(train_min, 3),
        "peak_mem_gb": round(mem, 3) if mem == mem else None,
        "device": str(DEVICE),
        "epochs": cfg.epochs,
        "dataset": f"CASIA-HWDB2-line validation, freq>={MIN_FREQ} ({len(keep_chars)} chars + UNK)",
    }
    print("\nFINAL_METRICS=" + json.dumps(m, ensure_ascii=False))
    print("\nSample predictions (first 6):")
    for i in range(min(6, len(preds))):
        ok = "✓" if preds[i] == gts[i] else "✗"
        print(f"  {ok} pred={preds[i][:40]!r}")
        print(f"     gt  ={gts[i][:40]!r}")
    return m


if __name__ == "__main__":
    main()
