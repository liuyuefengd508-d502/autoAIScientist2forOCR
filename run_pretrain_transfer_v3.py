"""
Run F v3: longer fine-tune (50 ep) + more synth images (200/class).
Builds on the successful v2 config (CER 0.4925).
If train.parquet is available it merges both splits automatically.
"""
from __future__ import annotations
import glob, collections, io, json, os, random, time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")

# ── 0. Load real data (val only, or val+train if train.parquet ready) ────────
def parquet_rows(path):
    import pyarrow.parquet as pq
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(512, columns=["text","image"]):
        df = batch.to_pandas()
        for _, r in df.iterrows():
            rows.append((r["image"]["bytes"], r["text"]))
    return rows

VAL_P   = glob.glob("./hf_cache/datasets--Teklia--CASIA-HWDB2-line/snapshots/*/data/validation.parquet")[0]
TRAIN_P = glob.glob("./hf_cache/datasets--Teklia--CASIA-HWDB2-line/snapshots/*/data/train.parquet")
TRAIN_P = TRAIN_P[0] if TRAIN_P else None

print(f"[data] loading validation ({os.path.getsize(VAL_P)/1e6:.0f} MB)…")
t0 = time.time()
all_rows = parquet_rows(VAL_P)
if TRAIN_P:
    print(f"[data] loading train ({os.path.getsize(TRAIN_P)/1e6:.0f} MB)…")
    all_rows += parquet_rows(TRAIN_P)
    print(f"[data] combined {len(all_rows)} lines in {time.time()-t0:.1f}s")
else:
    print(f"[data] {len(all_rows)} lines (val only) in {time.time()-t0:.1f}s")

# ── 1. Vocab ─────────────────────────────────────────────────────────────────
freq = collections.Counter(c for _, t in all_rows for c in t)
MIN_FREQ = 20
KEEP_CHARS = sorted(c for c, n in freq.items() if n >= MIN_FREQ)
VOCAB = set(KEEP_CHARS)
UNK_CH = "\x00"
ALL_CHARS = KEEP_CHARS + [UNK_CH]
BLANK = 0
CHAR2IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}
IDX2CHAR  = {i+1: c for i, c in enumerate(ALL_CHARS)}
NUM_CLASSES = len(ALL_CHARS) + 1
coverage = sum(v for c,v in freq.items() if c in VOCAB) / sum(freq.values())
print(f"[vocab] {len(KEEP_CHARS)} chars (freq≥{MIN_FREQ}), coverage={coverage*100:.1f}%, classes={NUM_CLASSES}")

def normalize(t):
    return "".join(c if c in VOCAB else UNK_CH for c in t)

random.seed(0)
random.shuffle(all_rows)
n_train = int(len(all_rows) * 0.9)
TRAIN_REAL = [(b, normalize(t)) for b, t in all_rows[:n_train] if t]
VAL_REAL   = [(b, normalize(t)) for b, t in all_rows[n_train:] if t]
print(f"[split] train={len(TRAIN_REAL)} val={len(VAL_REAL)}")

# ── 2. Fonts ──────────────────────────────────────────────────────────────────
FONT_DIRS = ["/System/Library/Fonts", "/Library/Fonts",
             os.path.expanduser("~/Library/Fonts")]

def find_chinese_fonts(n=8):
    out, test = [], "的"
    for d in FONT_DIRS:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if not f.lower().endswith((".ttf",".otf",".ttc")): continue
            fp = os.path.join(d, f)
            try:
                font = ImageFont.truetype(fp, 32)
                b = font.getbbox(test)
                if b[2]-b[0] > 5:
                    out.append(fp)
                    if len(out) >= n: return out
            except Exception: pass
    return out

FONTS = find_chinese_fonts()
print(f"[fonts] {len(FONTS)}: {[os.path.basename(f) for f in FONTS]}")

def render_char(ch, fp, size=40, img_size=48):
    font = ImageFont.truetype(fp, size)
    img  = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    b = font.getbbox(ch)
    x = (img_size - (b[2]-b[0]))//2 - b[0]
    y = (img_size - (b[3]-b[1]))//2 - b[1]
    draw.text((x, y), ch, fill=0, font=font)
    arr = np.array(img, dtype=np.float32)/255.0
    arr += np.random.normal(0, 0.05, arr.shape)
    arr -= np.random.uniform(0, 0.1)
    return np.clip(arr, 0, 1)

class SynthCharDS(Dataset):
    def __init__(self, chars, n_per_class=200, seed=0):
        rng = random.Random(seed)
        self.samples = []
        for lbl, c in enumerate(chars):
            for _ in range(n_per_class):
                fp   = rng.choice(FONTS)
                size = rng.randint(26, 42)
                arr  = render_char(c, fp, size)
                arr  = (arr - 0.5) / 0.5
                self.samples.append((arr.astype(np.float32), lbl))
        rng.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        a, l = self.samples[i]
        return torch.from_numpy(a).unsqueeze(0), l

# ── 3. Model ──────────────────────────────────────────────────────────────────
class SharedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = 32
        self.layers = nn.Sequential(
            nn.Conv2d(1,c,3,1,1), nn.GELU(), nn.MaxPool2d(2,2),
            nn.Conv2d(c,c*2,3,1,1), nn.GELU(), nn.MaxPool2d(2,2),
            nn.Conv2d(c*2,c*4,3,1,1), nn.GELU(),
            nn.Conv2d(c*4,c*4,3,1,1), nn.GELU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(c*4,c*8,3,1,1), nn.GELU(),
            nn.Conv2d(c*8,c*8,3,1,1), nn.GELU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(c*8,c*8,(3,1),1,0), nn.GELU(),
        )
    def forward(self, x): return self.layers(x)

class ClfHead(nn.Module):
    def __init__(self, in_ch=256, n=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_ch, n)
    def forward(self, f): return self.fc(self.pool(f).flatten(1))

class CRNNHead(nn.Module):
    def __init__(self, in_ch=256, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.rnn  = nn.LSTM(in_ch, hidden, 2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden*2, num_classes)
    def forward(self, f):
        s, _ = self.rnn(f.squeeze(2).transpose(1,2))
        return self.head(s)

# ── 4. Real-line helpers ──────────────────────────────────────────────────────
def img_to_tensor(b, h=48):
    img = Image.open(io.BytesIO(b)).convert("L")
    if img.height != h:
        w = max(8, int(img.width*h/img.height))
        img = img.resize((w,h), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)/255.0
    return torch.from_numpy((a-0.5)/0.5).unsqueeze(0)

class RealLineDS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        b, t = self.rows[i]
        return img_to_tensor(b), torch.tensor([CHAR2IDX[c] for c in t], dtype=torch.long)

def collate(batch):
    xs, ys = zip(*batch)
    H, mw = xs[0].shape[1], max(x.shape[-1] for x in xs)
    pad = torch.full((len(xs),1,H,mw), -1.0)
    for i,x in enumerate(xs): pad[i,:,:,:x.shape[-1]] = x
    return pad, torch.cat(ys), torch.tensor([len(y) for y in ys], dtype=torch.long)

# ── 5. Decode / CER ───────────────────────────────────────────────────────────
def greedy(logits):
    pred = logits.argmax(-1).detach().cpu().numpy()
    out = []
    for row in pred:
        s, prev = [], -1
        for v in row:
            if v != prev and v != BLANK: s.append(IDX2CHAR.get(int(v),""))
            prev = v
        out.append("".join(s))
    return out

def cer(preds, gts):
    def ed(a,b):
        if a==b: return 0
        if not a: return len(b)
        if not b: return len(a)
        dp = list(range(len(b)+1))
        for ca in a:
            prev, dp[0] = dp[0], dp[0]+1
            for j,cb in enumerate(b,1):
                cur=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,prev+(ca!=cb)); prev=cur
        return dp[-1]
    return sum(ed(p,g) for p,g in zip(preds,gts))/max(1,sum(len(g) for g in gts))

# ── 6. Config ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    pretrain_epochs: int  = 6
    pretrain_lr:     float = 5e-4
    finetune_epochs: int  = 50
    finetune_lr:     float = 5e-4
    batch_size:      int  = 32
    n_per_class:     int  = 200     # more synth images (was 80)
    grad_clip:       float = 5.0

# ── 7. Main ───────────────────────────────────────────────────────────────────
def main(cfg=None):
    cfg = cfg or Config()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"[cfg] {cfg}")

    # ── Phase 1: synthetic pretraining ────────────────────────────────────────
    print(f"\n{'='*55}\nPhase 1: synthetic isolated-char pretraining\n{'='*55}")
    n_synth = len(KEEP_CHARS)*cfg.n_per_class
    print(f"[synth] {len(KEEP_CHARS)}×{cfg.n_per_class} = {n_synth} images …")
    t0 = time.time()
    ds = SynthCharDS(KEEP_CHARS, cfg.n_per_class)
    print(f"[synth] done {time.time()-t0:.1f}s")
    nv = len(ds)//10
    tr_s, va_s = torch.utils.data.random_split(ds,[len(ds)-nv,nv],
                     generator=torch.Generator().manual_seed(0))
    tr_dl = DataLoader(tr_s, cfg.batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_s, cfg.batch_size, shuffle=False, num_workers=0)

    cnn = SharedCNN().to(DEVICE)
    clf = ClfHead(256, len(KEEP_CHARS)).to(DEVICE)
    opt1= torch.optim.AdamW(list(cnn.parameters())+list(clf.parameters()), lr=cfg.pretrain_lr)
    for ep in range(cfg.pretrain_epochs):
        cnn.train(); clf.train()
        losses, c, n = [], 0, 0
        for x,y in tr_dl:
            x,y = x.to(DEVICE),y.to(DEVICE)
            logits = clf(cnn(x)); loss = F.cross_entropy(logits,y)
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cnn.parameters())+list(clf.parameters()), cfg.grad_clip)
            opt1.step()
            losses.append(float(loss.detach().cpu()))
            c+=(logits.argmax(1)==y).sum().item(); n+=y.size(0)
        cnn.eval(); clf.eval(); vc=vt=0
        with torch.no_grad():
            for x,y in va_dl:
                x,y=x.to(DEVICE),y.to(DEVICE)
                vc+=(clf(cnn(x)).argmax(1)==y).sum().item(); vt+=y.size(0)
        print(f"[pre ep {ep:2d}] loss={np.mean(losses):.4f}  tr={c/n*100:.1f}%  val={vc/vt*100:.1f}%", flush=True)

    # ── Phase 2: CTC fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*55}\nPhase 2: CTC fine-tune on {len(TRAIN_REAL)} real lines\n{'='*55}")
    head  = CRNNHead(256, 256, NUM_CLASSES).to(DEVICE)
    opt2  = torch.optim.AdamW([
        {"params": cnn.parameters(),  "lr": cfg.finetune_lr*0.05},
        {"params": head.parameters(), "lr": cfg.finetune_lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.finetune_epochs, eta_min=1e-5)
    ctc   = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    tr_dl2 = DataLoader(RealLineDS(TRAIN_REAL), cfg.batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
    va_dl2 = DataLoader(RealLineDS(VAL_REAL),   cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    t1 = time.time()
    for ep in range(cfg.finetune_epochs):
        cnn.train(); head.train()
        losses = []
        for x,tgt,tl in tr_dl2:
            x = x.to(DEVICE)
            lp = F.log_softmax(head(cnn(x)), -1).transpose(0,1)
            T  = torch.full((x.size(0),), lp.size(0), dtype=torch.long)
            loss = ctc(lp.cpu(), tgt, T, tl)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cnn.parameters())+list(head.parameters()), cfg.grad_clip)
            opt2.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        avg = np.mean(losses)

        # quick val CER every 5 epochs
        if ep % 5 == 4 or ep == cfg.finetune_epochs-1:
            cnn.eval(); head.eval(); ps,gs = [],[]
            with torch.no_grad():
                for x,tgt,tl in va_dl2:
                    ps.extend(greedy(head(cnn(x.to(DEVICE)))))
                    off=0
                    for L in tl.tolist():
                        gs.append("".join(IDX2CHAR.get(int(c),"") for c in tgt[off:off+L])); off+=L
            val_cer = cer(ps, gs)
            print(f"[ft ep {ep:2d}] ctc={avg:.4f}  val_cer={val_cer:.4f}  ({time.time()-t1:.0f}s)", flush=True)
        else:
            print(f"[ft ep {ep:2d}] ctc={avg:.4f}  ({time.time()-t1:.0f}s)", flush=True)

    # ── Final eval ────────────────────────────────────────────────────────────
    cnn.eval(); head.eval(); ps,gs = [],[]
    with torch.no_grad():
        for x,tgt,tl in va_dl2:
            ps.extend(greedy(head(cnn(x.to(DEVICE)))))
            off=0
            for L in tl.tolist():
                gs.append("".join(IDX2CHAR.get(int(c),"") for c in tgt[off:off+L])); off+=L

    val_cer = cer(ps, gs)
    try:   mem = torch.mps.driver_allocated_memory()/1024**3
    except: mem = float("nan")
    n_p = sum(p.numel() for p in list(cnn.parameters())+list(head.parameters()))

    m = {
        "cer": round(val_cer,4),
        "params_M": round(n_p/1e6,3),
        "num_classes": NUM_CLASSES,
        "n_train_real": len(TRAIN_REAL),
        "n_val_real": len(VAL_REAL),
        "n_synth": n_synth,
        "finetune_min": round((time.time()-t1)/60,2),
        "peak_mem_gb": round(mem,3) if mem==mem else None,
        "device": str(DEVICE),
        "pretrain_ep": cfg.pretrain_epochs,
        "finetune_ep": cfg.finetune_epochs,
        "dataset": "synth-pretrain→CASIA-HWDB2 (val" + ("+train" if TRAIN_P else "") + ")",
    }
    print("\nFINAL_METRICS=" + json.dumps(m, ensure_ascii=False))
    print("\nSample predictions:")
    for i in range(min(6,len(ps))):
        ok = "✓" if ps[i]==gs[i] else "✗"
        print(f"  {ok} pred={ps[i][:50]!r}")
        print(f"     gt  ={gs[i][:50]!r}")
    return m

if __name__ == "__main__":
    main()
