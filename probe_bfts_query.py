"""Replicate the EXACT BFTS first call (_define_global_metrics) using the actual code path."""
import time
from ai_scientist.treesearch.backend import query

# This mirrors parallel_agent._define_global_metrics
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

print(f"task_desc len: {len(task_desc)} chars")
t0 = time.time()
try:
    response = query(
        system_message=prompt,
        user_message=None,
        model="gpt-5.4-mini",
        temperature=1.0,
        max_tokens=12000,
    )
    print(f"OK in {time.time()-t0:.1f}s")
    print("--- response (first 400 chars) ---")
    print(str(response)[:400])
except Exception as e:
    print(f"FAIL in {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:400]}")
