"""Build SFT data and the vanilla reference costs (for the GRPO reward).

Rejection sampling: for each training question, sample k DA traces from the model *with
masking on*, and keep the ones that are correct, well formed, and focused on the chunk that
actually has the needle. Sampling under the mask matters: a trace that only worked because
the model could still see the document in <local> is exactly what we don't want to copy.

    python -m src.sample_traces --model Qwen/Qwen3-4B --n-questions 600 --k 8 --out data/sft_kept.jsonl

--vanilla-ref instead runs one greedy vanilla pass per question and records how many tokens
it attended. rewards.py uses that as the reference cost.

    python -m src.sample_traces --vanilla-ref --n-questions 400 --out data/grpo_train.jsonl

--ideal skips the model entirely and writes templated "perfect" traces (synth.ideal_trace).
That's plan B if the base model almost never succeeds under masking and rejection sampling
comes back nearly empty. It teaches the format but it's not the model's own reasoning, so
note it in the write-up if we end up using it.
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.data import read_jsonl
from src.decode import generate
from src.paths import expand
from src.generate import build_prompt, load_model, pick_device
from src.metrics import attended_tokens, dense_attended_tokens, exact_match
from src.parse import check_trace, extract_answer, replay
from src.synth import ideal_trace, ruler_key


def pick_questions(rows, n, seed, buckets=None):
    if buckets:
        rows = [r for r in rows if r["bucket"] in buckets]
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:n]


def keep(ex, text, token_texts, layout, needle):
    m = replay(token_texts, layout.n_chunks)
    ok, _ = check_trace(text, layout.n_chunks)
    focused = {i for ids in m.focused for i in ids}
    return (exact_match(extract_answer(text), ex["answer"]) and ok and bool(focused & set(needle or []))
            and m.focus_successes >= 1), attended_tokens(layout, m.modes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter")
    ap.add_argument("--data", default="data/niah_train.jsonl")
    ap.add_argument("--out", required=True, help="a bare filename goes to $LTF_ROOT/data/")
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--buckets", nargs="+")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-keep", type=int, default=2, help="keep at most this many good traces per question")
    ap.add_argument("--nomask", action="store_true", help="sample without masking (not recommended, see docstring)")
    ap.add_argument("--vanilla-ref", action="store_true")
    ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = pick_questions(read_jsonl(args.data), args.n_questions, args.seed, args.buckets)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} questions to do ({len(done)} already in {out})")

    if args.ideal:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        rng = random.Random(args.seed)
        with open(out, "a") as f:
            for ex in rows:
                p, needle = build_prompt(tok, ex, "da")
                if not needle:
                    continue
                what, key = ruler_key(ex["question"]) if ex["task"] != "synth" else ("code", ex["key"])
                f.write(json.dumps({**ex, "target": ideal_trace(key, ex["answer"], needle[0], rng, what=what),
                                    "source": "ideal_template", "needle_chunks": needle}) + "\n")
        return

    model, tok = load_model({"model": args.model, "adapter": args.adapter, "dtype": "auto"}, pick_device())
    t0, n_kept, n_tried = time.time(), 0, 0
    with open(out, "a") as f:
        for qi, ex in enumerate(rows):
            if args.vanilla_ref:
                p, _ = build_prompt(tok, ex, "vanilla")
                tr = generate(model, tok, [p], masked=False, max_new_tokens=args.max_new_tokens)[0]
                ex = {**ex, "vanilla_attended": dense_attended_tokens(p.layout.n_prompt, len(tr.token_ids)),
                      "vanilla_correct": exact_match(extract_answer(tr.text), ex["answer"])}
                f.write(json.dumps(ex) + "\n")
                f.flush()
                continue

            p, needle = build_prompt(tok, ex, "da")
            traces = generate(model, tok, [p] * args.k, masked=not args.nomask, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, seed=args.seed * 7919 + qi)
            good = []
            for tr in traces:
                texts = [tok.decode([i]) for i in tr.token_ids]
                ok, att = keep(ex, tr.text, texts, p.layout, needle) if tr.finished else (False, 0)
                if ok:
                    good.append((att, tr))
            n_tried += len(traces)
            # among the good ones prefer the cheapest (fewest attended tokens), and drop exact duplicates
            good.sort(key=lambda x: x[0])
            seen = set()
            for att, tr in good:
                if tr.text in seen or len(seen) >= args.max_keep:
                    continue
                seen.add(tr.text)
                f.write(json.dumps({**ex, "target": tr.text, "attended": att, "source": "rejection_sampling",
                                    "needle_chunks": needle}) + "\n")
                n_kept += 1
            f.flush()
            print(f"[{qi + 1}/{len(rows)}] {len(good)}/{len(traces)} good   kept so far {n_kept}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not args.vanilla_ref:
        print(f"kept {n_kept} traces from {n_tried} samples")


if __name__ == "__main__":
    main()
