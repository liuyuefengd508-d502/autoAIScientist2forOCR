"""Run with reduced charset (~30 chars) and tuned LR to verify the CRNN can actually learn."""
import json, time, random, importlib
import ai_scientist.ideas.handwritten_archive_ocr as seed

# Override charset to a small subset before instantiating dataset
SMALL_CHARSET = list("一二三四五六七八九十的是了我不人有这个上中") + list("0123456789")
seed.CHARSET = SMALL_CHARSET
seed.CHAR2IDX = {c: i + 1 for i, c in enumerate(SMALL_CHARSET)}
seed.IDX2CHAR = {i + 1: c for i, c in enumerate(SMALL_CHARSET)}
seed.NUM_CLASSES = len(SMALL_CHARSET) + 1
print(f"[override] charset size = {len(SMALL_CHARSET)} (from full {len(seed.COMMON_HANZI)+22})")

# Re-bind on TinyCRNN init: keep model default (will use NUM_CLASSES at instantiation)
t0 = time.time()
# Patch TinyCRNN default arg
seed.TinyCRNN.__init__.__defaults__ = (seed.NUM_CLASSES, 192)

# Also use shorter sequences (2-6 chars)
orig_dataset = seed.SyntheticHandwrittenLines
class ShortDS(orig_dataset):
    def __init__(self, n_samples, **kw):
        kw.setdefault("min_len", 2)
        kw.setdefault("max_len", 6)
        super().__init__(n_samples, **kw)
seed.SyntheticHandwrittenLines = ShortDS

m = seed.main(seed.Config(
    seed=0,
    n_train=5000,
    n_val=500,
    batch_size=32,
    epochs=20,
    lr=2e-4,
    grad_clip=5.0,
))
print(f"\n[total wall-clock: {(time.time()-t0)/60:.1f} min]")
print("FINAL:", json.dumps(m, indent=2, ensure_ascii=False))
