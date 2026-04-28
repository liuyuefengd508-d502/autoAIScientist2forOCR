"""Replicate the exact first BFTS call (_define_global_metrics) to find why it 502s."""
import time, json, openai

client = openai.OpenAI(max_retries=0, timeout=180)

# Mimic what backend_openai.query does in fallback mode for func_spec
schema = {
    "type": "object",
    "properties": {
        "metric_name": {"type": "string"},
        "maximize": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["metric_name", "maximize", "description"],
    "additionalProperties": False,
}
schema_str = json.dumps(schema, ensure_ascii=False)
sys_msg = (
    "You are a careful research scientist.\n\n"
    "You MUST respond with a single valid JSON object only "
    "(no prose, no markdown fences). The JSON object MUST match this "
    "JSON schema for the function `define_metric` (`define the global metric`):\n"
    + schema_str
)

# Test with progressively larger user_message to see where it breaks
for size_label, mult in [("small", 1), ("medium", 100), ("large", 400), ("xlarge", 1000)]:
    user_msg = "Decide the metric for OCR. " * mult
    for attempt in range(2):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=1.0,
                max_tokens=12000,   # this is what BFTS passes
            )
            print(f"[{size_label} attempt {attempt}] OK {time.time()-t0:.1f}s "
                  f"prompt_toks={r.usage.prompt_tokens} out={r.choices[0].message.content[:80]!r}")
            break
        except Exception as e:
            print(f"[{size_label} attempt {attempt}] FAIL {time.time()-t0:.1f}s "
                  f"{type(e).__name__}: {str(e)[:200]}")

# Now try without max_tokens=12000 (use default)
print("\n--- no max_tokens ---")
for size_label, mult in [("medium", 100), ("large", 400)]:
    user_msg = "Decide the metric for OCR. " * mult
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        print(f"[{size_label} no-maxtok] OK {time.time()-t0:.1f}s prompt_toks={r.usage.prompt_tokens}")
    except Exception as e:
        print(f"[{size_label} no-maxtok] FAIL {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:200]}")
