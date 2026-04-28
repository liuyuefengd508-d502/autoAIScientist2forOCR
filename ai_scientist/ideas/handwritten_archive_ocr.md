# Title: Beyond DenseNet+TCN: Compute-Aware Improvements to End-to-End Chinese Handwritten Archive OCR

## Keywords
Chinese handwritten OCR, end-to-end text spotting, archive digitization, lightweight transformer, model compression, structured pruning, knowledge distillation, page-level recognition, layout analysis, synthetic handwriting generation, Apple Silicon MPS, on-device deep learning

## TL;DR
Build on the recently published end-to-end Chinese handwritten archive OCR framework of Zhao (2025) — which combines multi-scale text detection, progressive scale shrink post-processing, a DenseNet+TCN recognizer, and magnitude pruning — and propose **compute-aware improvements** that can be **trained and evaluated entirely on a single Apple M5 Pro GPU (24 GB unified memory, PyTorch MPS)** using only **publicly available** Chinese handwriting corpora. The goal is to find architectural, augmentation, and compression changes that improve accuracy / efficiency trade-offs reported by Zhao (AR 88.91%, CR 89.36%, F1 95.15; after compression: −86.58% params with only −2.83% AR / −2.18% CR), under far stricter compute conditions.

## Abstract

### Background
Handwritten archives carry rich social, cultural and historical information, and their digitization is a national priority. Compared with printed-text OCR, Chinese handwritten archive recognition is significantly harder because of:
1. **Complex page layout** — mixed vertical/horizontal lines, ruled tables, marginalia, stamps, multi-column flow, dense and sparse line spacing, large character-scale variability.
2. **Severe writing-background noise** — paper aging, ink bleed-through, fading, stains, foxing, ruled grids, low-contrast scans, skew/warping.
3. **Large character vocabulary** — Chinese has thousands of common characters; archives often contain rare or variant glyphs.
4. **Diverse writing styles** — different writers, periods and pen types; cursive ligatures and stroke variations.
5. **Limited public data** — labeled handwritten archive corpora are scarce relative to printed text or English HTR.
6. **Heavy parameter budgets of end-to-end systems** — current SOTA pipelines have tens to hundreds of millions of parameters, hindering deployment on edge / archive workstations / laptops.

### Reference work and gaps
**Zhao (2025)** proposes a two-part end-to-end framework on a self-built Chinese handwritten archive dataset:
- **Detection**: a multi-scale feature-enhanced detector with a *progressive scale shrink* post-processing algorithm to separate dense neighbouring lines and resist background noise.
- **Recognition**: an improved **DenseNet + Temporal Convolutional Network (TCN)** model trained with CTC loss, with transfer learning from public + synthetic data to the archive set.
- **Lightweight variant**: a lightweight backbone with a "dimension+scale" feature-enhancement module, a text rectification post-processing module (mask correction + TPS + affine), a CNN + lightweight Transformer recognizer, and **magnitude iterative pruning** for compression.
- **Reported numbers**: AR 88.91%, CR 89.36%, detection F1 95.15; after compression −86.58% parameters with only −2.83% / −2.18% AR/CR drop.

Identified **gaps / open questions** that we want to investigate:
- The DenseNet+TCN choice predates modern lightweight conformer / linear-attention recognizers. Are there tiny recognizers (≤15M params) that match its CER while being faster on commodity / on-device accelerators (especially Apple MPS)?
- Magnitude pruning is unstructured and rarely yields wall-clock speedup on real hardware. Does **structured pruning + post-training quantization (int8 dynamic, fp16)** give similar params reduction with measurable speedup on MPS?
- Page synthesis from "single-character images + corpus" is reported but only mildly evaluated. How does it compare to **physics-inspired degradation augmentation** (paper-texture overlay, ink-bleed simulation, low-contrast aging, stroke perturbation) when the recognizer is small?
- Progressive scale shrink is heuristic. Can a **lightweight Gaussian/heatmap-based line representation** (like CenterNet-style) achieve similar separation of dense lines with fewer ops?
- Cross-writer / cross-period generalization is not explicitly measured. **Leave-one-writer-out (LOWO)** and **leave-one-source-out (LOSO)** evaluation reveals real archive deployment robustness.
- Decoder-side improvements (radical/structure constrained beam search, lightweight character n-gram LM rescoring) are largely orthogonal and cheap; are they better ROI than enlarging the recognizer?

### Research questions
We aim to answer, through small-scale, controlled experiments on a single Apple Silicon laptop:
- **RQ1 — Recognizer architecture**: At ≤15M parameters and ≤512 sequence length, which family wins on Chinese handwritten lines: (a) tiny CRNN+CTC, (b) DenseNet+TCN-tiny (Zhao-style), (c) Conformer-CTC-tiny, (d) distilled small Transformer encoder–decoder?
- **RQ2 — Compression**: Does **structured channel pruning + int8/ fp16 PTQ** match or beat Zhao's magnitude pruning, in (params reduction, AR drop, MPS wall-clock latency)?
- **RQ3 — Synthesis vs degradation augmentation**: For the same training budget, does adding physics-inspired **degradation augmentation** to corpus+single-char synthesis improve real-archive CER more than synthesis alone?
- **RQ4 — Detection lightening**: Replace the multi-scale module with **MobileNetV3-Small / MobileViT-XS + FPN-lite**; does the progressive scale shrink post-processing still deliver F1 gains, or can it be replaced with a learned gather-and-merge head?
- **RQ5 — Cross-writer / cross-source generalization**: Under LOWO and LOSO splits, which design choices retain accuracy best?
- **RQ6 — Decoder-side cheap wins**: How much CER does a small (≤5MB) character n-gram LM + radical-constrained beam search add to a fixed recognizer, compared to growing the recognizer by the same parameter budget?

### Datasets (publicly available, ≤1 GB each, no DRM)
- **CASIA-HWDB 1.0 / 1.1** (offline isolated handwritten Chinese characters, ~3.7M samples; subsample to 50–200k for fast iteration).
- **CASIA-HWDB 2.x** (handwritten text lines; subsample to a few thousand pages).
- **ICDAR 2013 Chinese Handwriting** (lines/pages, small).
- **MTHv2** subset (Chinese historical handwritten manuscripts).
- **Synthetic page-level set** generated locally via PIL using system Chinese fonts + text from public Wikipedia/news corpora, plus on-the-fly degradation augmentation.

If a dataset is unavailable or > 1 GB, the experiment should fall back to **on-the-fly synthetic Chinese line images** rendered from system fonts (`Songti`, `Heiti`, `STKaiti`, `STFangsong`, etc.) with degradation augmentation — this guarantees runnability without network downloads.

### Mandatory hardware/software constraints (must be respected by every experiment)
- **Device**: a single Apple Silicon GPU via PyTorch MPS. Use:
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- **Forbidden**: any of `.cuda()`, `torch.cuda.*`, `cuda` device strings, `nccl`, `apex`, `bitsandbytes`, `flash-attn`, `xformers`, `triton`. If a library auto-imports CUDA, do not use it.
- **Model size**: total trainable parameters ≤ **150M** (prefer ≤ 50M for fast iteration, ≤ 15M for the recognizer when isolating it).
- **Peak GPU memory**: ≤ **18 GB** (leave headroom for OS / unified memory pressure).
- **Wall-clock per experiment**: ≤ **60 minutes** total. Use small subsets (5k–50k line images), batch ≤ 16, ≤ 10 epochs unless explicitly justified.
- **Network**: do not download datasets > 1 GB or models > 500 MB during a run. Use cached HuggingFace `datasets` / `transformers` snapshots when possible.
- **Mixed precision**: prefer `torch.autocast(device_type="mps", dtype=torch.float16)` where stable; fall back to fp32 if numerically unstable.
- **Reproducibility**: every reported number must be **mean ± std over ≥ 3 seeds** unless wall-clock budget forbids it (then report ≥ 1 seed and explicitly state).
- **Required reported metrics for every experiment**:
  - **CER (character error rate)**, **AR (accuracy rate)**, **CR (correct rate)** for recognition;
  - **Detection F1 / precision / recall** if a detection stage is present;
  - **Params (M)**, **MAC/FLOPs (G)**, **MPS wall-clock training time (min)**, **MPS inference latency per page (ms)**, **peak unified-memory usage (GB)**.

### What this project is NOT
- Not training large foundation models from scratch.
- Not aiming to beat Zhao's absolute numbers on his private archive set (we don't have it).
- Not pursuing pure SOTA on huge HTR benchmarks.
- Not a survey paper.

### Expected contribution
The deliverable is an **empirical, compute-aware comparison study** that gives practitioners a clear answer to: *"If I can only afford a laptop-class accelerator and public Chinese handwriting data, which combination of recognizer family + compression + synthesis/augmentation + decoding gives the best accuracy / latency / memory trade-off for archive OCR?"* — and where the **failure modes** are (worth surfacing in the spirit of "I Can't Believe It's Not Better").
