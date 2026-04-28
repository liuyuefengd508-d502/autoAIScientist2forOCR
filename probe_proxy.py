"""Probe the proxy with controlled prompt sizes; sanity-check gpt-5.4-mini before BFTS."""
import os
import time
import openai


def run():
    client = openai.OpenAI(max_retries=0, timeout=180)
    print("base_url:", client.base_url)

    # A. small plain chat
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "say pong"}],
            max_tokens=10,
        )
        print(f"[A small chat mini] OK {time.time()-t0:.1f}s -> {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"[A] FAIL {type(e).__name__}: {str(e)[:250]}")

    # B. small + json_object
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": 'Reply with one JSON object: {"ok":true}'},
                {"role": "user", "content": "hi"},
            ],
            response_format={"type": "json_object"},
            max_tokens=20,
        )
        print(f"[B small json mini] OK {time.time()-t0:.1f}s -> {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"[B] FAIL {type(e).__name__}: {str(e)[:250]}")

    # C. medium ~2k toks + json_object on mini
    big = "Design an OCR experiment step by step. " * 200
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": 'Reply JSON: {"plan":"brief 1-line plan"}'},
                {"role": "user", "content": big},
            ],
            response_format={"type": "json_object"},
            max_tokens=120,
        )
        print(f"[C medium ~2k mini] OK {time.time()-t0:.1f}s toks={r.usage.prompt_tokens}")
    except Exception as e:
        print(f"[C] FAIL after {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:250]}")

    # D. large ~6k toks + json_object on mini (closer to BFTS reality)
    bigger = big * 3
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": 'Reply JSON: {"plan":"brief 1-line plan"}'},
                {"role": "user", "content": bigger},
            ],
            response_format={"type": "json_object"},
            max_tokens=120,
        )
        print(f"[D large ~6k mini] OK {time.time()-t0:.1f}s toks={r.usage.prompt_tokens}")
    except Exception as e:
        print(f"[D] FAIL after {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:250]}")

    # E. very large ~12k toks + json_object on mini (BFTS upper bound)
    huge = big * 6
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": 'Reply JSON: {"plan":"brief 1-line plan"}'},
                {"role": "user", "content": huge},
            ],
            response_format={"type": "json_object"},
            max_tokens=120,
        )
        print(f"[E very large ~12k mini] OK {time.time()-t0:.1f}s toks={r.usage.prompt_tokens}")
    except Exception as e:
        print(f"[E] FAIL after {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:250]}")


if __name__ == "__main__":
    run()
