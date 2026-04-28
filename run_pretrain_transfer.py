"""
3-phase pipeline: synthetic-char pretraining → transfer → real-line fine-tune.

Phase 1 : generate ~120K synthetic isolated-char images (font + noise).
Phase 2 : pre-train TinyCRNN's CNN backbone as a 1421-class classifier.
Phase 3 : plug CNN weights into CTC TinyCRNN; fine-tune on real CASIA-HWDB2 lines.
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")

# ─────────────────────────────────────────────
# 0.  Vocab from real CASIA-HWDB2-line parquet
# ─────────────────────────────────────────────
PARQUET_PATH = glob.glob(
    "./hf_cache/datasets--Teklia--CASIA-HWDB2-line/snapshots/*/data/validation.parquet"
)[0]


def load_real_rows():
    import pyarrow.parquet as pq
    rows = []
    for batch in pq.ParquetFile(PARQUET_PATH).iter_batches(512, columns=["text", "image"]):
        df = batch.to_pandas()
        for _, r in df.iterrows():
            rows.append((r["image"]["bytes"], r["text"]))
    return rows


print("[data] loading real parquet …")
t0 = time.time()
real_rows = load_real_rows()
print(f"[data] {len(real_rows)} lines in {time.time()-t0:.1f}s")

freq = collections.Counter(c for _, t in real_rows for c in t)
MIN_FREQ = 20
KEEP_CHARS = sorted(c for c, n in freq.items() if n >= MIN_FREQ)
VOCAB = set(KEEP_CHARS)
UNK_CH = "\x00"
ALL_CHARS = KEEP_CHARS + [UNK_CH]   # UNK last
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALL_CHARS)}
IDX2CHAR  = {i + 1: c for i, c in enumerate(ALL_CHARS)}
NUM_CLASSES = len(ALL_CHARS) + 1   # +1 for CTC blank
print(f"[vocab] {len(KEEP_CHARS)} known chars (freq≥{MIN_FREQ}) + UNK → {NUM_CLASSES} CTC classes")


def normalize(t):
    return "".join(c if c in VOCAB else UNK_CH for c in t)


random.seed(0)
random.shuffle(real_rows)
n_train = int(len(real_rows) * 0.85)
TRAIN_REAL = [(b, normalize(t)) for b, t in real_rows[:n_train] if t]
VAL_REAL   = [(b, normalize(t)) for b, t in real_rows[n_train:] if t]
print(f"[split] real train={len(TRAIN_REAL)} val={len(VAL_REAL)}")


# ─────────────────────────────────────────────
# 1.  Synthetic isolated-char dataset
# ─────────────────────────────────────────────
FONT_DIRS = [
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]


def find_chinese_fonts(max_fonts=6):
    """Find system fonts that can render common Chinese chars."""
    candidates = []
    test_char = "的"
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            fp = os.path.join(d, f)
            try:
                font = ImageFont.truetype(fp, 32)
                bbox = font.getbbox(test_char)
                if bbox[2] - bbox[0] > 5:
                    candidates.append(fp)
                    if len(candidates) >= max_fonts:
                        return candidates
            except Exception:
                pass
    return candidates


FONTS = find_chinese_fonts()
print(f"[fonts] found {len(FONTS)} Chinese fonts: {[os.path.basename(f) for f in FONTS]}")
if not FONTS:
    raise RuntimeError("No Chinese font found on system. Install a CJK font.")


def render_char(char: str, font_path: str, size: int = 40, img_size: int = 48) -> np.ndarray:
    """Render one character, add noise/degradation, return float32 [0,1] HW array."""
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_size - w) // 2 - bbox[0]
    y = (img_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    arr = np.array(img, dtype=np.float32) / 255.0
    # noise + brightness
    arr += np.random.normal(0, 0.05, arr.shape)
    arr -= np.random.uniform(0, 0.1)
    arr = np.clip(arr, 0, 1)
    return arr


class SyntheticCharDataset(Dataset):
    """N_PER_CLASS synthetic images per char in KEEP_CHARS."""

    def __init__(self, chars, n_per_class=80, seed=0):
        rng = random.Random(seed)
        self.samples = []  # (arr_bytes, label_idx)
        for label, c in enumerate(chars):
            for k in range(n_per_class):
                font = rng.choice(FONTS)
                # vary font size 28-40
                size = rng.randint(28, 40)
                arr = render_char(c, font, size=size, img_size=48)
                # normalise to [-1, 1]
                arr = (arr - 0.5) / 0.5
                self.samples.append((arr.astype(np.float32), label))
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        return torch.from_numpy(arr).unsqueeze(0), label


# ─────────────────────────────────────────────
# 2.  Model: shared CNN + two heads
# ─────────────────────────────────────────────
class SharedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = 32
        self.layers = nn.Sequential(
            nn.Conv2d(1, c, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c*2, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(c*2, c*4, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c*4, c*4, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c*4, c*8, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c*8, c*8, 3, 1, 1), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(c*8, c*8, (3, 1), 1, 0), nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)   # (B, 256, 1, W') for line images


class ClassifierHead(nn.Module):
    """For isolated-char classification: global-avg-pool the spatial dim."""
    def __init__(self, in_ch=256, num_chars=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_ch, num_chars)

    def forward(self, feat):
        return self.fc(self.pool(feat).flatten(1))


class CRNNHead(nn.Module):
    """For CTC line recognition: LSTM over time axis."""
    def __init__(self, in_ch=256, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.rnn  = nn.LSTM(in_ch, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, feat):
        # feat: (B, C, 1, W')
        seq = feat.squeeze(2).transpose(1, 2)   # (B, W', C)
        out, _ = self.rnn(seq)
        return self.head(out)                    # (B, T, num_classes)


# ─────────────────────────────────────────────
# 3.  Real-line DataLoader helpers
# ─────────────────────────────────────────────
def img_to_tensor(img_bytes: bytes, h: int = 48) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    if img.height != h:
        w = max(8, int(img.width * h / img.height))
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5).unsqueeze(0)


class RealLineDS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        b, t = self.rows[idx]
        x = img_to_tensor(b, 48)
        y = torch.tensor([CHAR2IDX[c] for c in t], dtype=torch.long)
        return x, y


def collate_lines(batch):
    xs, ys = zip(*batch)
    H, max_w = xs[0].shape[1], max(x.shape[-1] for x in xs)
    pad = torch.full((len(xs), 1, H, max_w), -1.0)
    for i, x in enumerate(xs):
        pad[i, :, :, :x.shape[-1]] = x
    return pad, torch.cat(ys), torch.tensor([len(y) for y in ys], dtype=torch.long)


# ─────────────────────────────────────────────
# 4.  Greedy CTC decode / CER
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# 5.  Main training loop
# ─────────────────────────────────────────────
@dataclass
class Config:
    pretrain_epochs: int = 5       # phase 1: isolated-char classification
    pretrain_lr: float = 5e-4
    finetune_epochs: int = 25      # phase 2: CTC on real lines — more epochs
    finetune_lr: float = 5e-4      # higher head LR (was 1e-4)
    batch_size: int = 32
    n_per_class: int = 80          # synthetic images per char
    grad_clip: float = 5.0


def main(cfg: Config | None = None):
    cfg = cfg or Config()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    # ── Phase 1: generate synthetic data + pre-train CNN ──────────────────
    print(f"\n{'='*60}\nPhase 1: Synthetic isolated-char pretraining\n{'='*60}")
    print(f"[synth] generating {len(KEEP_CHARS)}×{cfg.n_per_class} = "
          f"{len(KEEP_CHARS)*cfg.n_per_class} images …")
    t0 = time.time()
    synth_ds = SyntheticCharDataset(KEEP_CHARS, n_per_class=cfg.n_per_class)
    print(f"[synth] done in {time.time()-t0:.1f}s")

    n_val_s = len(synth_ds) // 10
    synth_train, synth_val = torch.utils.data.random_split(
        synth_ds, [len(synth_ds) - n_val_s, n_val_s],
        generator=torch.Generator().manual_seed(0)
    )
    synth_train_dl = DataLoader(synth_train, cfg.batch_size, shuffle=True, num_workers=0)
    synth_val_dl   = DataLoader(synth_val,   cfg.batch_size, shuffle=False, num_workers=0)

    cnn  = SharedCNN().to(DEVICE)
    clf  = ClassifierHead(in_ch=256, num_chars=len(KEEP_CHARS)).to(DEVICE)
    opt1 = torch.optim.AdamW(list(cnn.parameters()) + list(clf.parameters()), lr=cfg.pretrain_lr)

    for ep in range(cfg.pretrain_epochs):
        cnn.train(); clf.train()
        losses, correct, total = [], 0, 0
        for x, y in synth_train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = clf(cnn(x))
            loss = F.cross_entropy(logits, y)
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cnn.parameters())+list(clf.parameters()), cfg.grad_clip)
            opt1.step()
            losses.append(float(loss.detach().cpu()))
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
        # val acc
        cnn.eval(); clf.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in synth_val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vc += (clf(cnn(x)).argmax(1) == y).sum().item()
                vt += y.size(0)
        print(f"[pretrain ep {ep:2d}] loss={np.mean(losses):.4f}  "
              f"train_acc={correct/total*100:.1f}%  val_acc={vc/vt*100:.1f}%", flush=True)

    # ── Phase 2: fine-tune CTC on real lines ──────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Transfer + CTC fine-tune on real CASIA-HWDB2\n{'='*60}")

    crnn_head = CRNNHead(in_ch=256, hidden=256, num_classes=NUM_CLASSES).to(DEVICE)
    # CNN weights transferred from pretraining (already on DEVICE)
    opt2 = torch.optim.AdamW(
        [{"params": cnn.parameters(), "lr": cfg.finetune_lr * 0.05},  # very slow for pretrained CNN
         {"params": crnn_head.parameters(), "lr": cfg.finetune_lr}]    # full LR for fresh LSTM head
    )
    # Keep LR reasonably high for longer; eta_min prevents decay to zero
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg.finetune_epochs, eta_min=1e-5
    )
    ctc   = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    train_dl = DataLoader(RealLineDS(TRAIN_REAL), cfg.batch_size, shuffle=True,  collate_fn=collate_lines, num_workers=0)
    val_dl   = DataLoader(RealLineDS(VAL_REAL),   cfg.batch_size, shuffle=False, collate_fn=collate_lines, num_workers=0)

    t1 = time.time()
    for ep in range(cfg.finetune_epochs):
        cnn.train(); crnn_head.train()
        losses = []
        for x, tgt, tgt_len in train_dl:
            x = x.to(DEVICE)
            feat = cnn(x)                            # (B, 256, 1, W')
            logits = crnn_head(feat)                 # (B, T, C)
            lp = F.log_softmax(logits, dim=-1).transpose(0, 1)
            T  = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
            loss = ctc(lp.cpu(), tgt, T, tgt_len)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cnn.parameters())+list(crnn_head.parameters()), cfg.grad_clip)
            opt2.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        print(f"[finetune ep {ep:2d}] ctc_loss={np.mean(losses):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t1:.0f}s)", flush=True)

    finetune_min = (time.time()-t1)/60

    # ── Evaluate ────────────────────────────────────────────────────────────
    cnn.eval(); crnn_head.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, tgt, tgt_len in val_dl:
            preds.extend(greedy_decode(crnn_head(cnn(x.to(DEVICE)))))
            off = 0
            for L in tgt_len.tolist():
                gts.append("".join(IDX2CHAR.get(int(c), "") for c in tgt[off:off+L]))
                off += L

    val_cer = cer(preds, gts)
    try:
        mem = torch.mps.driver_allocated_memory() / 1024**3
    except Exception:
        mem = float("nan")

    n_params = sum(p.numel() for p in list(cnn.parameters()) + list(crnn_head.parameters()))
    m = {
        "cer": round(val_cer, 4),
        "params_M": round(n_params / 1e6, 3),
        "num_classes": NUM_CLASSES,
        "n_train_real": len(TRAIN_REAL),
        "n_val_real": len(VAL_REAL),
        "n_synth": len(KEEP_CHARS) * cfg.n_per_class,
        "finetune_time_min": round(finetune_min, 3),
        "peak_mem_gb": round(mem, 3) if mem == mem else None,
        "device": str(DEVICE),
        "pretrain_epochs": cfg.pretrain_epochs,
        "finetune_epochs": cfg.finetune_epochs,
        "dataset": "synthetic-pretrain → CASIA-HWDB2-line real fine-tune",
    }
    print("\nFINAL_METRICS=" + json.dumps(m, ensure_ascii=False))
    print("\nSample predictions (first 8):")
    for i in range(min(8, len(preds))):
        ok = "✓" if preds[i] == gts[i] else "✗"
        print(f"  {ok} pred={preds[i][:40]!r}")
        print(f"     gt  ={gts[i][:40]!r}")
    return m


if __name__ == "__main__":
    main()
