"""Sort failed DA traces into buckets so we know *why* DA loses accuracy.

This is a first pass, not a replacement for reading traces. The categories are checked in
order and each failed trace lands in the first one that fits:

  ran_out          hit max_new_tokens without finishing
  no_answer        finished but never wrote <answer>...</answer>
  broken_tags      tags don't parse cleanly (unclosed, nested, bad chunk ids)
  never_focused    answered without a single successful <focus>. under masking that means
                   the answer came from memory, i.e. a guess (the paper's "hallucinate in local")
  wrong_chunk      focused, but never on the chunk that holds the needle
  bad_extraction   focused the right chunk, but the value written inside <focus> isn't the answer
  lost_after_focus extracted the right value inside <focus>, then answered something else anyway

    python -m src.failures runs/baseline_qwen3_4b_4k --arm da --show 3
"""

import argparse
import collections
import random
import re
from pathlib import Path

from src.data import read_jsonl
from src.metrics import normalize
from src.parse import check_trace, extract_answer

CATEGORIES = ["ran_out", "no_answer", "broken_tags", "never_focused", "wrong_chunk",
              "bad_extraction", "lost_after_focus"]

FOCUS_BLOCK = re.compile(r'<focus\s+magic_chunks="([^"]*)">(.*?)</focus>', re.DOTALL)


def categorize(trace):
    """trace: one row of traces.jsonl. Returns a category, or None if it was correct."""
    text, gold = trace["text"], trace["gold"]
    pred = extract_answer(text)
    if pred is not None and normalize(pred) == normalize(gold):
        return None
    if not trace.get("finished", True):
        return "ran_out"
    if pred is None:
        return "no_answer"
    n_chunks = len(trace["layout"]["chunk_spans"])
    ok, _ = check_trace(text, n_chunks)
    if not ok:
        return "broken_tags"
    blocks = FOCUS_BLOCK.findall(text)
    if not blocks:
        return "never_focused"
    needle = set(trace.get("needle_chunks") or [])
    on_needle = []
    for ids, body in blocks:
        try:
            got = {int(x) for x in re.split(r"[,\s]+", ids.strip()) if x}
        except ValueError:
            continue
        if got & needle:
            on_needle.append(body)
    if not on_needle:
        return "wrong_chunk"
    if not any(normalize(gold) in normalize(b) for b in on_needle):
        return "bad_extraction"
    return "lost_after_focus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arm", default="da")
    ap.add_argument("--show", type=int, default=0, help="print this many example traces per category")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(Path(args.run) / "traces.jsonl") if r["arm"] == args.arm]
    by_cat = collections.defaultdict(list)
    for r in rows:
        c = categorize(r)
        if c:
            by_cat[c].append(r)
    n_fail = sum(len(v) for v in by_cat.values())
    print(f"{args.arm}: {len(rows)} traces, {n_fail} wrong")
    for c in CATEGORIES:
        n = len(by_cat[c])
        print(f"  {c:18s} {n:4d}  ({n / max(n_fail, 1):5.1%} of failures)")
    rng = random.Random(0)
    for c in CATEGORIES:
        for r in rng.sample(by_cat[c], min(args.show, len(by_cat[c]))):
            print(f"\n----- {c}  {r['id']}  gold={r['gold']}  needle={r.get('needle_chunks')}")
            print(r["text"][:1500])


if __name__ == "__main__":
    main()
