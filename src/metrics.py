"""Per-example metrics: accuracy, tag adherence, decode steps, attended tokens.

Attended tokens is the efficiency number. At every decode step we count how many KV positions
the model is allowed to look at, then add that up over the whole trace:

    step i (1-based) has seq_len = n_prompt + i
    global (or vanilla):  everything, i.e. seq_len
    focus on chunks K:    sink + len(chunks in K) + (seq_len - question_start)
    local / answer:       sink + (seq_len - question_start)

(seq_len - question_start) is "the question + instructions + everything written so far",
which is always visible. This is the same thing as the formula in our project notes, just
written in positions instead of lengths.

The off-by-one convention (step i uses n_prompt + i, and the mode BEFORE token i was fed)
matches the authors' ledger exactly. tests/test_parity.py checks that against their fixtures.

We count one layer. The paper counts per global-attention layer, but that's a constant
factor so percent changes come out the same.
"""

import math

from src.parse import FOCUS, GLOBAL, LOCAL, check_trace, extract_answer, replay


def visible_count(layout, mode, focus_ids, seq_len):
    """How many positions are visible at a step with this seq_len."""
    if mode == GLOBAL:
        return seq_len
    tail = seq_len - layout.question_start
    if mode == LOCAL:
        return layout.sink + tail
    spans = sum(layout.chunk_spans[i - 1][1] - layout.chunk_spans[i - 1][0] for i in set(focus_ids))
    return layout.sink + spans + tail


def attended_tokens(layout, modes):
    """modes: list of (mode, focus_ids), one per generated token (TagMachine.modes)."""
    return sum(visible_count(layout, m, ids, layout.n_prompt + i) for i, (m, ids) in enumerate(modes, start=1))


def dense_attended_tokens(n_prompt, n_steps):
    """What a trace of n_steps costs with nothing hidden. Vanilla uses this, and it's also the
    denominator for the 'how much did DA save on this exact trace' ratio."""
    return n_steps * n_prompt + n_steps * (n_steps + 1) // 2


def normalize(s):
    return " ".join(s.strip().lower().split())


def exact_match(pred, gold):
    if pred is None:
        return False
    return normalize(pred) == normalize(gold)


def score_trace(trace_text, token_texts, layout, gold, needle_chunks=None, is_da=True):
    """Everything we track for one generated answer, as a flat dict (one CSV row).

    token_texts: the generated text split per token (so we can replay the state machine).
    needle_chunks: chunk ids that contain the answer, if we know them (NIAH: we do).
    """
    pred = extract_answer(trace_text)
    n_steps = len(token_texts)
    row = {
        "pred": pred,
        "gold": gold,
        "correct": exact_match(pred, gold),
        "has_answer": pred is not None,
        "decode_steps": n_steps,
        "n_prompt": layout.n_prompt,
        "dense_attended": dense_attended_tokens(layout.n_prompt, n_steps),
    }
    if not is_da:
        row["attended"] = row["dense_attended"]
        return row

    m = replay(token_texts, layout.n_chunks)
    ok, problems = check_trace(trace_text, layout.n_chunks)
    steps_in = {GLOBAL: 0, FOCUS: 0, LOCAL: 0}
    for mode, _ in m.modes:
        steps_in[mode] += 1
    focused_ids = {i for ids in m.focused for i in ids}

    row.update({
        "attended": attended_tokens(layout, m.modes),
        "parse_ok": ok,
        "parse_problems": "; ".join(problems),
        # the paper's "focus success": closed focus blocks / focus attempts
        "focus_attempts": m.focus_attempts,
        "focus_successes": m.focus_successes,
        "focus_success_rate": m.focus_successes / m.focus_attempts if m.focus_attempts else math.nan,
        "focus_failures": sum(m.focus_failures.values()),
        "n_focus_blocks": len(m.focused),
        "chunks_focused": ",".join(str(i) for i in sorted(focused_ids)),
        "global_steps": steps_in[GLOBAL],
        "focus_steps": steps_in[FOCUS],
        "local_steps": steps_in[LOCAL],
        "n_chunks": layout.n_chunks,
    })
    if needle_chunks:
        # our own stricter thing: did it ever actually look at the right chunk?
        row["needle_hit"] = bool(focused_ids & set(needle_chunks))
        row["needle_chunks"] = ",".join(map(str, needle_chunks))
    return row


def summarize(rows):
    """Aggregate a list of score_trace rows into one dict. NaNs are skipped in means."""
    def mean(key):
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return sum(vals) / len(vals) if vals else math.nan

    out = {"n": len(rows)}
    for k in ["correct", "has_answer", "parse_ok", "needle_hit", "decode_steps", "focus_attempts",
              "focus_success_rate"]:
        out[k] = mean(k)
    att = sum(r["attended"] for r in rows)
    dense = sum(r["dense_attended"] for r in rows)
    out["attended_total"] = att
    out["attended_per_example"] = att / len(rows) if rows else math.nan
    # pooled focus success, like the paper: sum of successes / sum of attempts
    tries = sum(r.get("focus_attempts", 0) for r in rows)
    out["focus_success_pooled"] = sum(r.get("focus_successes", 0) for r in rows) / tries if tries else math.nan
    out["attention_ratio"] = att / dense if dense else math.nan
    return out
