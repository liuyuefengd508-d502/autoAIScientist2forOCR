"""See exactly what compile_prompt_to_md produces, then send it raw to the proxy."""
import time, openai
from ai_scientist.treesearch.backend.utils import compile_prompt_to_md

task_desc = open("ai_scientist/ideas/handwritten_archive_ocr.md").read()

prompt = {
    "Introduction": (
        "You are an AI researcher setting up experiments. "
        "Please propose meaningful evaluation metrics that will help analyze "
        "the performance and characteristics of solutions for this research task."
    ),
    "Research idea": task_desc,
    "Instructions": [
        "Propose a single evaluation metric that would be useful for analyzing the performance of solutions for this research task.",
        "Note: Validation loss will be tracked separately so you don't need to include it in your response.",
        "Format your response as a list containing:",
        "- name: The name of the metric",
        "- maximize: Whether higher values are better (true/false)",
        "- description: A brief explanation of what the metric measures"
        "Your list should contain only one metric.",
    ],
}

compiled = compile_prompt_to_md(prompt)
print(f"compiled prompt: {len(compiled)} chars")
print("---first 500 chars---")
print(compiled[:500])
print("---last 500 chars---")
print(compiled[-500:])
print("---")

client = openai.OpenAI(max_retries=0, timeout=180)
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "system", "content": compiled}],
        temperature=1.0,
        max_tokens=12000,
    )
    print(f"OK {time.time()-t0:.1f}s prompt_toks={r.usage.prompt_tokens} out_chars={len(r.choices[0].message.content)}")
    print("--- response (first 300 chars) ---")
    print(r.choices[0].message.content[:300])
except Exception as e:
    print(f"FAIL {time.time()-t0:.1f}s {type(e).__name__}: {str(e)[:300]}")
