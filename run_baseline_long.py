"""Run the seed baseline with more aggressive settings to actually train down."""
import json, time
from ai_scientist.ideas.handwritten_archive_ocr import main, Config

t0 = time.time()
m = main(Config(
    seed=0,
    n_train=5000,
    n_val=500,
    batch_size=32,
    epochs=15,
    lr=5e-4,
    grad_clip=5.0,
))
print(f"\n[total wall-clock: {(time.time()-t0)/60:.1f} min]")
print("FINAL:", json.dumps(m, indent=2, ensure_ascii=False))
