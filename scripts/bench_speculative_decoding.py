"""Research spike for issue #29: does the pinned Qwen2.5-Coder-1.5B speed up chat
generation as a speculative-decoding draft model for a larger Qwen2.5-Coder?

Standalone and throwaway — not imported by the server. Run directly:

    python scripts/bench_speculative_decoding.py

Loads both models once, then for each prompt runs a baseline (no draft) and a
speculative (draft_model=) generation at temperature=0 and reports tok/s, speedup,
and the draft acceptance rate.

TARGET_REPO defaults to the 7B (fits alongside the draft in ~6GB free RAM on this
machine). The pair this issue actually cares about — 1.5B draft / 32B target, the
model already in Ember's registry — needs ~14GB free; rerun with
TARGET_REPO="mlx-community/Qwen2.5-Coder-32B-Instruct-3bit" when that much is free.
"""

import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

TARGET_REPO = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DRAFT_REPO = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
MAX_TOKENS = 200
NUM_DRAFT_TOKENS = 4

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Explain what a race condition is and give a short example in Go.",
    "Fix the bug: `def avg(xs): return sum(xs) / len(xs)` fails on an empty list. Show the fix.",
    "Refactor this to avoid the nested loop: "
    "`for i in range(len(a)):\\n    for j in range(len(a)):\\n        if a[i] == a[j] and i != j: found = True`",
    "What's the difference between a process and a thread? Answer in two sentences.",
]


def _chat_prompt(tok, text):
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, add_generation_prompt=True)


def _run(model, tok, prompt_ids, draft_model=None):
    sampler = make_sampler(temp=0.0)
    last = None
    accepted = 0
    total = 0
    t0 = time.perf_counter()
    for r in stream_generate(
        model,
        tok,
        mx.array(prompt_ids),
        max_tokens=MAX_TOKENS,
        draft_model=draft_model,
        num_draft_tokens=NUM_DRAFT_TOKENS if draft_model is not None else None,
        sampler=sampler,
    ):
        last = r
        total += 1
        if r.from_draft:
            accepted += 1
    wall = time.perf_counter() - t0
    tps = last.generation_tokens / wall if last else 0.0
    acceptance = accepted / total if total else 0.0
    return tps, acceptance, last.generation_tokens if last else 0


def main():
    print(f"Loading target: {TARGET_REPO}")
    model, tok = load(TARGET_REPO)
    print(f"Loading draft:  {DRAFT_REPO}")
    draft_model, draft_tok = load(DRAFT_REPO)

    assert tok.vocab_size == draft_tok.vocab_size, (
        f"vocab mismatch: target={tok.vocab_size} draft={draft_tok.vocab_size}"
    )

    rows = []
    for prompt in PROMPTS:
        ids = _chat_prompt(tok, prompt)
        base_tps, _, base_toks = _run(model, tok, ids, draft_model=None)
        spec_tps, acceptance, spec_toks = _run(model, tok, ids, draft_model=draft_model)
        speedup = (spec_tps / base_tps - 1.0) * 100 if base_tps else 0.0
        rows.append((prompt[:40], base_tps, spec_tps, speedup, acceptance))
        print(
            f"[{prompt[:40]!r}] baseline={base_tps:.1f} tok/s "
            f"speculative={spec_tps:.1f} tok/s ({speedup:+.1f}%) "
            f"acceptance={acceptance:.0%} "
            f"(base_toks={base_toks}, spec_toks={spec_toks})"
        )

    print("\n| Prompt | baseline tok/s | speculative tok/s | speedup | acceptance |")
    print("|---|--:|--:|--:|--:|")
    for prompt, base_tps, spec_tps, speedup, acceptance in rows:
        print(
            f"| {prompt} | {base_tps:.1f} | {spec_tps:.1f} | {speedup:+.1f}% | {acceptance:.0%} |"
        )


if __name__ == "__main__":
    main()
