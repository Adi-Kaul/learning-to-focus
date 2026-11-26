"""Turning DA modes into attention masks.

Two flavours:

- decode_mask_row: one row of 0/1 for "which positions can the next token look at". The
  generation loop (decode.py) builds one of these every step.

- training_mask: a full (L x L) mask for a whole prompt+trace, so a single forward pass
  sees exactly what step-by-step masked generation would have seen. This is what lets us
  train under the same masking we evaluate with. Without it, SFT/GRPO would train the model
  with the whole document visible in <local> mode, which is exactly the habit we're trying to
  break (it learns to "remember" values it never focused on, then falls over once the mask
  is real).

A detail that matters: which mode applies to which row. The token at position p is the
INPUT that predicts token p+1. So the row for p uses the mode that token p+1 was generated
in. Prompt rows are plain causal (the prompt is prefilled with nothing hidden, same as the
paper's engine).
"""

import torch

from src.parse import FOCUS, GLOBAL


def visible_positions(layout, mode, focus_ids, seq_len):
    """Boolean vector of length seq_len: True = can attend."""
    vis = torch.ones(seq_len, dtype=torch.bool)
    if mode == GLOBAL:
        return vis
    vis[layout.sink:layout.question_start] = False
    if mode == FOCUS:
        for i in focus_ids:
            a, b = layout.chunk_spans[i - 1]
            vis[a:b] = True
    return vis


def training_mask(layout, gen_modes, final_mode=(GLOBAL, ()), dtype=torch.bool):
    """Mask for teacher forcing over prompt + T generated tokens.

    gen_modes: list of (mode, focus_ids), one per generated token, i.e. TagMachine.modes.
    final_mode: the machine's state after the last token (only affects the last row, which
    doesn't predict anything we train on, but keep it honest anyway).

    Returns a (L, L) bool tensor, True = attend, with causality already baked in.
    """
    P, T = layout.n_prompt, len(gen_modes)
    L = P + T
    mask = torch.ones(L, L, dtype=torch.bool).tril_()
    # row P-1+k (k = 1..T) is the input that generates token k+1.
    # token k+1 was generated in gen_modes[k] (0-based), the last row uses final_mode
    for k in range(1, T + 1):
        mode, ids = gen_modes[k] if k < T else final_mode
        if mode == GLOBAL:
            continue
        row = P - 1 + k
        mask[row, layout.sink:layout.question_start] = False
        if mode == FOCUS:
            for i in ids:
                a, b = layout.chunk_spans[i - 1]
                mask[row, a:b] = True
    return mask.to(dtype)


def mode_runs(gen_modes):
    """Compress [(mode, ids)] * T into [(start, end, mode, ids)] runs. Handy for debugging
    and for building masks for long traces without looping row by row in python."""
    runs = []
    for i, m in enumerate(gen_modes):
        if runs and runs[-1][2:] == m:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, *m])
    return [tuple(r) for r in runs]
