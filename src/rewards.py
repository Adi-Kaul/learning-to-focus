"""Reward for GRPO.

    R = r_correct + r_format + lam * r_correct * r_efficiency

  r_correct     1 if the first <answer> matches exactly (strip + lowercase), else 0
  r_format      +0.2 if the trace is well formed (parse.check_trace), -0.2 otherwise
  r_efficiency  1 - attended / reference, clipped to [0, 1]
  lam           how much we care about reading less. start at 0.3

Efficiency only pays out when the answer is right, so "don't read anything and guess"
can't win.

The reference cost: what the *vanilla* prompt cost the base model on the same question
(attended tokens of its vanilla trace). That's the honest comparison, it's what the paper
reports, and it means a DA trace that rambles on for 500 tokens in <local> still gets
punished, because every extra token adds to its attended count while the reference stays
put. Run `python -m src.sample_traces --vanilla-ref` once to fill it in for the training set.
If it's missing we fall back to the dense cost of the same trace (the "attention ratio"),
which does NOT punish rambling. That fallback is printed loudly.
"""

from src.metrics import attended_tokens, dense_attended_tokens, exact_match
from src.parse import check_trace, extract_answer, replay

FORMAT_BONUS = 0.2


def efficiency(attended, reference):
    if not reference:
        return 0.0
    return min(1.0, max(0.0, 1.0 - attended / reference))


def reward_parts(text, token_texts, layout, gold, reference=None, lam=0.3):
    """Returns a dict with every piece, so we can log them separately (handy for spotting hacks)."""
    correct = 1.0 if exact_match(extract_answer(text), gold) else 0.0
    ok, _ = check_trace(text, layout.n_chunks)
    fmt = FORMAT_BONUS if ok else -FORMAT_BONUS
    m = replay(token_texts, layout.n_chunks)
    att = attended_tokens(layout, m.modes)
    ref = reference or dense_attended_tokens(layout.n_prompt, len(token_texts))
    eff = efficiency(att, ref)
    return {
        "reward": correct + fmt + lam * correct * eff,
        "r_correct": correct,
        "r_format": fmt,
        "r_efficiency": eff,
        "attended": att,
        "reference": ref,
        "used_fallback_ref": reference is None,
        "n_focus": len(m.focused),
        "n_tokens": len(token_texts),
    }
