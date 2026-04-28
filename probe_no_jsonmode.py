"""Probe a plain (no response_format) chat completion with BFTS-like params."""
import time, openai, json

client = openai.OpenAI(max_retries=0, timeout=180)

# Long idea+seed prompt similar to what BFTS sends to _define_global_metrics
big = """You are an AI researcher setting up experiments. Propose meaningful evaluation metrics that will help analyze the performance and characteristics of solutions for this research task.

# Title: Beyond DenseNet+TCN: Compute-Aware Improvements to End-to-End Chinese Handwritten Archive OCR

## Keywords
Chinese handwritten OCR, end-to-end text spotting, archive digitization, lightweight transformer, model compression, structured pruning, knowledge distillation, page-level recognition, layout analysis, synthetic handwriting generation, Apple Silicon MPS, on-device deep learning

## TL;DR
Build on the recently published end-to-end Chinese handwritten archive OCR framework of Zhao (2025) which combines multi-scale text detection, progressive scale shrink post-processing, a DenseNet+TCN recognizer, and magnitude pruning, and propose compute-aware improvements that can be trained and evaluated entirely on a single Apple M5 Pro GPU (24 GB unified memory, PyTorch MPS) using only publicly available Chinese handwriting corpora.

""" * 10  # ~ several K tokens

for label, params in [
    ("plain", dict(model="gpt-5.4-mini", messages=[{"role":"user","content":big}], temperature=1.0, max_tokens=12000)),
    ("plain+seed", dict(model="gpt-5.4-mini", messages=[{"role":"user","content":big}], temperature=1.0, max_tokens=12000, seed=0)),
    ("plain+sys", dict(model="gpt-5.4-mini", messages=[{"role":"system","content":"You are a careful researcher."},{"role":"user","content":big}], temperature=1.0, max_tokens=12000)),
    # JSON mode but no func_spec (mimic of no-fallback path with mini)
    ("plain", dict(model="gpt-5.4-mini", messages=[{"role":"user","content":big}], temperature=1.0, max_tokens=4096)),
]:
    t0 = time.time()
    try:
        r = client.chat.completions.create(**params)
        dt = time.time()-t0
        print(f"[{label} max_tok={params.get('max_tokens')}] OK {dt:.1f}s prompt_toks={r.usage.prompt_tokens} out_chars={len(r.choices[0].message.content)}")
    except Exception as e:
        dt = time.time()-t0
        msg = str(e)[:300]
        print(f"[{label} max_tok={params.get('max_tokens')}] FAIL {dt:.1f}s {type(e).__name__}: {msg}")
