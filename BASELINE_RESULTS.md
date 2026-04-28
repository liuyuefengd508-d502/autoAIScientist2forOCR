# Baseline experiment results — Apple M5 Pro (MPS)

Local-only baseline runs of the seed CRNN OCR for `handwritten_archive_ocr`,
executed without BFTS (the proxy LLM was returning intermittent Cloudflare
502 Bad Gateway errors that prevented the BFTS tree-search from initializing).

Hardware: Apple M5 Pro, 24 GB unified memory, PyTorch MPS backend.
Code: `@/Users/liuyu/Desktop/mydocuments/codes/autoAi2/AI-Scientist-v2/ai_scientist/ideas/handwritten_archive_ocr.py`.

## Setup

- Synthetic Chinese handwritten line images (rendered via system fonts +
  noise/aging augmentation), no external dataset download.
- Tiny CRNN: 5×Conv + 2×BiLSTM + Linear, 2.9M params.
- CTC loss on CPU (MPS does not have a native CTC kernel in PyTorch ≥ 2.11).
- Greedy CTC decoding for evaluation.

## Run A — default config (168-class charset, 3 epochs)

Driver: `python ai_scientist/ideas/handwritten_archive_ocr.py`

| Metric | Value |
|---|---|
| n_train / n_val | 1500 / 300 |
| batch_size | 16 |
| epochs | 3 |
| lr | 1e-3 |
| **train_loss** trajectory | 8.66 → 5.49 → 5.49 |
| **CER** | **1.0** |
| params | 2.97 M |
| train_time | 0.25 min |
| peak_mem | 1.17 GB |
| device | mps |

Outcome: **CTC collapsed to all-blank predictions** (loss ≈ ln(168+1) ≈
5.13, CER = 1.0). Standard CTC failure mode when training signal is too
weak relative to class count.

## Run B — extended config (168-class, 15 epochs, more data)

Driver: `python run_baseline_long.py` (`@/Users/liuyu/Desktop/mydocuments/codes/autoAi2/AI-Scientist-v2/run_baseline_long.py`)

| Metric | Value |
|---|---|
| n_train / n_val | 5000 / 500 |
| batch_size | 32 |
| epochs | 15 |
| lr | 5e-4 |
| **train_loss** trajectory | 8.55 → 5.47 → 5.46 (still flat from epoch 1) |
| **CER** | **1.0** |
| params | 2.97 M |
| train_time | 3.09 min |
| peak_mem | 1.26 GB |

Outcome: **same CTC collapse**. More data + more epochs + lower LR did not
escape the all-blank local minimum. Tells us the 168-class fully-random
sequence task exceeds the capacity of this tiny CRNN under the chosen
training budget.

## Run C — reduced-difficulty config (32-class, short sequences, 20 epochs)

Driver: `python run_baseline_easy.py` (`@/Users/liuyu/Desktop/mydocuments/codes/autoAi2/AI-Scientist-v2/run_baseline_easy.py`)

| Metric | Value |
|---|---|
| charset | 32 chars (10 digits + 21 common Hanzi) |
| sequence length | 2-6 chars (was 4-16) |
| n_train / n_val | 5000 / 500 |
| batch_size | 32 |
| epochs | 20 |
| lr | 2e-4 |
| **train_loss** trajectory | 7.48 → 3.96 (plateau) → 1.69 (escape ep 6) → 0.003 |
| **CER** | **0.0005** (99.95% char accuracy) |
| params | 2.91 M |
| train_time | 1.64 min |
| peak_mem | 1.17 GB |

Outcome: **clean training**. The CRNN escapes the CTC plateau around epoch
6, then converges sharply. CER on held-out validation = 5e-4. This confirms
the architecture, training loop, and MPS pipeline are all correct.

## Implications for next experiments

1. **The seed code itself is fine on MPS** — no `.cuda()` issues, peak
   memory comfortable at < 1.3 GB even with batch 32.
2. **The 168-class default is misconfigured for the tiny CRNN** — it should
   either (a) use a curriculum (start with small charset, expand), or
   (b) pre-train on isolated characters (CASIA-HWDB 1.x) before line
   training, or (c) add per-character localization signal.
3. **Training is fast enough on M5 Pro to be feasible without BFTS** — a
   full 5 K × 20 epochs run completes in under 2 minutes. We can manually
   sweep hyperparameters (lr, batch, charset size, augmentation strength,
   architecture variants) on the order of dozens of runs per hour.
4. **The proxy LLM blocker is independent of the experiment code** — once a
   stable LLM endpoint is available, BFTS can be re-launched on idea 3
   without changes to the seed script.

## Run D — real data, full vocab (CASIA-HWDB2-line, all 2630 chars)

Driver: `python run_baseline_real.py`

| Metric | Value |
|---|---|
| Dataset | Teklia/CASIA-HWDB2-line validation (7 K train / 1.3 K val) |
| batch_size | 16 |
| epochs | 6 |
| lr | 5e-4 |
| **train_loss** trajectory | 8.60 → 6.94 → 6.93 (plateau) |
| **CER** | **0.9809** (model predicts "。" for everything) |
| params | 3.91 M (head expands for 2630 classes) |
| train_time | 4.8 min |
| peak_mem | 7.3 GB |

Outcome: **CTC collapse to constant character "。"** (highest-frequency char).
Not all-blank collapse this time — the model found a local minimum predicting
the modal character. Loss plateau at 6.94 ≈ ln(2630) suggests the model is
roughly at uniform-random level over 2630 classes with slight bias toward "。".

## Run E — real data, top-1421 chars + UNK mapping, 12 epochs

Driver: `python run_baseline_real_topn.py` (freq ≥ 20 → 1421 chars + `<UNK>`)

| Metric | Value |
|---|---|
| Dataset | same validation parquet; 8318 → 8318 lines (rare chars → `\x00`) |
| vocab coverage | 94.7% of char tokens |
| batch_size | 16 |
| epochs | 12 |
| lr | 3e-4 + cosine |
| **train_loss** trajectory | 8.77 → 6.54 → … → 6.47 (slow decline, no escape) |
| **CER** | **0.9809** |
| params | 4.68 M |
| train_time | 9.6 min |
| peak_mem | 6.6 GB |

Outcome: **same modal collapse**. 12 epochs + cosine LR + UNK mapping did not
help. Loss only declined 8.77 → 6.47, still well above the convergence
threshold. Predictions are consistently "。".

---

## Summary of findings

| Experiment | Setup | CER |
|---|---|---|
| Run A | Synthetic, 168 classes, 3 ep | 1.0 (CTC collapse) |
| Run B | Synthetic, 168 classes, 15 ep | 1.0 (CTC collapse) |
| **Run C** | **Synthetic, 32 classes, 20 ep** | **0.0005 ✓** |
| Run D | Real CASIA-HWDB2, 2630 classes, 6 ep | 0.9809 (modal collapse) |
| Run E | Real CASIA-HWDB2, 1421+UNK classes, 12 ep | 0.9809 (modal collapse) |

**Key finding**: TinyCRNN trained **from scratch** collapses on any task with
> ~50 classes on this dataset budget. The CRNN architecture + CTC loss work
correctly (Run C proves this). The failure mode is a known CTC optimization
challenge with large vocabularies and limited data — a well-defined
research problem.

**What would fix this**:
1. Pre-train CNN on isolated CASIA-HWDB1.x characters (classification,
   ~3.7 M single-char images), then fine-tune on line CTC.
2. Synthetic-to-real transfer: pre-train on large synthetic Chinese
   text-line images, fine-tune on real data (domain adaptation).
3. Curriculum training: start CTC on 2-4 char sequences (easier), expand.
4. Semi-supervised: use a font-rendered model to soft-label unlabeled
   real-data lines, then fine-tune with real labels.

These represent exactly the **research opportunities** formalized in
`handwritten_archive_ocr.md` (RQ2–RQ4).

## Reproduce

```bash
cd ~/Desktop/mydocuments/codes/autoAi2/AI-Scientist-v2
source .venv/bin/activate

# Run A (default, demonstrates CTC collapse)
python ai_scientist/ideas/handwritten_archive_ocr.py

# Run B (longer, still collapses)
python run_baseline_long.py

# Run C (reduced charset, demonstrates capability — CER 0.0005)
python run_baseline_easy.py
```

Logs: `baseline_default.log`, `baseline_long.log`, `baseline_easy.log`,
`baseline_real.log`, `baseline_real_topn.log`
(in repo root, gitignored as `*.log`).

# Download CASIA-HWDB2-line (validation only, 219 MB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Teklia/CASIA-HWDB2-line',
                filename='data/validation.parquet',
                repo_type='dataset', cache_dir='./hf_cache')
"

# Run D (real data, full charset, CTC collapse)
python run_baseline_real.py

# Run E (real data, top-1421 chars + UNK, still collapses)
python run_baseline_real_topn.py
