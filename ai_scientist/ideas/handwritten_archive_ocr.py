"""
Baseline seed for the BFTS tree search.

This is a minimal, MPS-friendly Chinese handwritten line recognizer:
  - synthetic line images rendered from system fonts (no dataset download needed)
  - tiny CRNN backbone (CNN -> BiLSTM -> linear -> CTC)
  - trains for a few hundred iterations on M5 Pro
  - reports CER, training wall-clock, peak memory, params

It is intentionally small and self-contained so it always runs end-to-end on a
24 GB Apple Silicon MPS GPU. The tree-search agents will mutate this file
(change the architecture, add augmentation, swap loss, etc.) — keep the
device handling, metric reporting, and entry point intact.

HARD CONSTRAINTS for any improved version:
  - No `.cuda()`, no `torch.cuda.*`, no `nccl`, no `apex`, no `bitsandbytes`,
    no `flash-attn`, no `xformers`, no `triton`. Use `torch.device("mps")`.
  - Total params <= 150M (prefer <= 50M).
  - Peak unified memory <= 18 GB.
  - Wall-clock <= 60 min.
  - Always print a final line that contains all metrics in JSON, e.g.:
      FINAL_METRICS={"cer": 0.42, "params_M": 1.8, "train_time_min": 0.7, ...}
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


# ----------------------------- device -----------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"[device] {DEVICE}")


# ----------------------------- data -----------------------------
# 用一个紧凑的中文字符集 (常用 ~500 字 + 数字 + 标点) 作为目标词表，便于快速训练。
# 真实研究中应替换为更大的字符集 / 训练集。
COMMON_HANZI = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年"
    "着就那和要她出也得里后自以会家可下而过天去能对小多然于"
    "心学么之都好看起发当没成只如事把还用第样道想作种开美总"
    "从无情已面最女但现前些所同日手又行意动方期它头经长儿回"
    "位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话"
    "常与活正感"
)
DIGITS = "0123456789"
PUNCT = "，。、；：？！“”‘’《》（）—…·"
CHARSET = list(dict.fromkeys(COMMON_HANZI + DIGITS + PUNCT))
BLANK = 0  # CTC blank
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for blank


def find_cn_font() -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Songti.ttc",
        "/Library/Fonts/STHeiti Light.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 32)
            except Exception:
                continue
    # last resort: PIL default (Latin only, will not render Hanzi)
    raise RuntimeError(
        "No usable Chinese font found. Add a .ttf/.ttc path to find_cn_font()."
    )


def render_line(text: str, font: ImageFont.FreeTypeFont, height: int = 48) -> Image.Image:
    """Render a line of text into a grayscale image with mild noise to mimic
    handwritten / aged archive scans."""
    pad_x, pad_y = 6, 6
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 2 * pad_x
    h = bbox[3] - bbox[1] + 2 * pad_y
    img = Image.new("L", (max(w, 32), max(h, height)), color=255)
    d = ImageDraw.Draw(img)
    d.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=0, font=font)
    # quick aging / noise
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 8.0, arr.shape)
    arr -= np.random.uniform(0, 25)  # global darkening (fading paper)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    # resize to fixed height while preserving aspect
    new_w = max(32, int(img.width * height / img.height))
    img = img.resize((new_w, height), Image.BILINEAR)
    return img


class SyntheticHandwrittenLines(Dataset):
    def __init__(self, n_samples: int = 2000, min_len: int = 4, max_len: int = 16, seed: int = 0):
        rng = random.Random(seed)
        self.font = find_cn_font()
        self.samples: list[tuple[Image.Image, str]] = []
        for _ in range(n_samples):
            n = rng.randint(min_len, max_len)
            text = "".join(rng.choice(CHARSET) for _ in range(n))
            img = render_line(text, self.font, height=48)
            self.samples.append((img, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, text = self.samples[idx]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # [-1, 1]
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        y = torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch)
    # pad widths
    H = xs[0].shape[1]
    max_w = max(x.shape[-1] for x in xs)
    pad_xs = torch.full((len(xs), 1, H, max_w), -1.0)
    widths = torch.zeros(len(xs), dtype=torch.long)
    for i, x in enumerate(xs):
        pad_xs[i, :, :, : x.shape[-1]] = x
        widths[i] = x.shape[-1]
    targets = torch.cat(ys)
    target_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return pad_xs, targets, target_lens, widths


# ----------------------------- model -----------------------------
class TinyCRNN(nn.Module):
    """Very small CRNN (~3-5M params) suitable for fast iteration on MPS."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 192):
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
        f = self.cnn(x)              # (B, C, 1, T)
        f = f.squeeze(2).transpose(1, 2)  # (B, T, C)
        f, _ = self.rnn(f)
        return self.head(f)          # (B, T, num_classes)


# ----------------------------- decode / metrics -----------------------------
def greedy_decode(logits: torch.Tensor) -> list[str]:
    pred = logits.argmax(-1).detach().cpu().numpy()  # (B, T)
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


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def compute_cer(preds: list[str], gts: list[str]) -> float:
    edits = sum(edit_distance(p, g) for p, g in zip(preds, gts))
    total = sum(len(g) for g in gts) or 1
    return edits / total


# ----------------------------- training loop -----------------------------
@dataclass
class Config:
    seed: int = 0
    n_train: int = 1500
    n_val: int = 300
    batch_size: int = 16
    epochs: int = 3
    lr: float = 1e-3
    grad_clip: float = 5.0


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"[cfg] {cfg}")

    train_ds = SyntheticHandwrittenLines(n_samples=cfg.n_train, seed=cfg.seed)
    val_ds = SyntheticHandwrittenLines(n_samples=cfg.n_val, seed=cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = TinyCRNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M, num_classes={NUM_CLASSES}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for batch_idx, (x, tgt, tgt_len, _w) in enumerate(train_loader):
            x = x.to(DEVICE)
            logits = model(x)                          # (B, T, C)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, C)
            T_len = torch.full((x.size(0),), log_probs.size(0), dtype=torch.long)
            # CTC needs CPU tensors on MPS for some PyTorch versions
            loss = ctc(log_probs.cpu(), tgt, T_len, tgt_len)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[epoch {ep}] train_loss={np.mean(losses):.4f}")

    train_time_min = (time.time() - t0) / 60.0

    # eval
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, tgt, tgt_len, _w in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(greedy_decode(logits))
            # rebuild gt strings
            offset = 0
            for L in tgt_len.tolist():
                gts.append("".join(IDX2CHAR[int(t)] for t in tgt[offset : offset + L]))
                offset += L
    cer = compute_cer(preds, gts)

    # peak memory (best-effort; MPS API limited)
    try:
        peak_mem_gb = torch.mps.driver_allocated_memory() / (1024 ** 3)
    except Exception:
        peak_mem_gb = float("nan")

    metrics = {
        "cer": round(cer, 4),
        "params_M": round(n_params / 1e6, 3),
        "train_time_min": round(train_time_min, 3),
        "peak_mem_gb": round(peak_mem_gb, 3) if peak_mem_gb == peak_mem_gb else None,
        "device": str(DEVICE),
        "n_train": cfg.n_train,
        "epochs": cfg.epochs,
    }
    print("FINAL_METRICS=" + json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    main()
