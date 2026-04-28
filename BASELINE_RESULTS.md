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

Logs: `baseline_default.log`, `baseline_long.log`, `baseline_easy.log`
(in repo root, gitignored as `*.log`).
