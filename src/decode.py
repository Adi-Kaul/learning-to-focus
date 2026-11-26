"""Generation loop with real DA masking, in plain Hugging Face Transformers.

Why not model.generate()? Because the mask has to change *between* decode steps based on what
the model just wrote, and generate() doesn't give us a clean hook for that. So this is a
hand-written loop: prefill the prompt once, then one forward per new token, with a fresh
attention mask each time.

It handles a batch of prompts with left padding. Each row has its own layout and its own
TagMachine, so row 0 can be in <focus> while row 1 is in <local>. Pads are just more zeros
in the mask.

The same loop also runs the unmasked arms (vanilla and "DA, no mask") so every arm goes
through identical code and only the mask differs.

This only does the *right* thing for models where every layer is full attention (Qwen3).
For hybrid models (Qwen3.5 with DeltaNet layers, Gemma with sliding window) the mask would
also be applied to layers the paper leaves alone, or ignored by them. Don't trust it there
without checking.
"""

from dataclasses import dataclass, field

import torch

from src.masking import visible_positions
from src.parse import TagMachine


@dataclass
class Trace:
    token_ids: list = field(default_factory=list)
    token_texts: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    text: str = ""
    finished: bool = False     # hit EOS (as opposed to running out of budget)
    eos_logprob: float = None  # logprob of the EOS token when it finished (GRPO wants it)
    machine: TagMachine = None


def _eos_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return set(ids)


def _sample(logits, temperature, top_p, generator):
    """logits: (B, V). Returns (token ids, logprob of the chosen token under the sampling dist)."""
    if temperature == 0:
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok = logits.argmax(dim=-1)
        return tok, logp.gather(1, tok[:, None]).squeeze(1)
    logits = logits.float() / temperature
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    if top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p   # always keeps the top token
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter_(1, idx, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    tok = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return tok, logp.gather(1, tok[:, None]).squeeze(1)


@torch.no_grad()
def generate(model, tokenizer, prompts, *, masked=True, max_new_tokens=1024,
             temperature=0.0, top_p=1.0, seed=0, force=None):
    """prompts: list of src.prompts.Prompt. Returns a list of Trace, same order.

    masked=False gives the "DA, no mask" arm for DA prompts (and is what you'd use for vanilla,
    where there's nothing to hide anyway).

    force: optional list of token id lists, one per prompt. Instead of sampling, feed these
    tokens and record their logprobs. Used by the tests to check this loop against the
    one-shot training mask, and handy for scoring a fixed trace under masking.
    """
    device = next(model.parameters()).device
    B = len(prompts)
    lens = [len(p.input_ids) for p in prompts]
    P = max(lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos = _eos_ids(model, tokenizer)
    gen = torch.Generator(device=device).manual_seed(seed)

    # left pad so every row's last prompt token is at the same column
    offsets = [P - n for n in lens]
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    attn = torch.zeros(B, P + max_new_tokens, dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, offsets[b]:] = torch.tensor(p.input_ids)
        attn[b, offsets[b]:P] = 1
    ids, attn = ids.to(device), attn.to(device)
    # real positions ignore the padding
    pos = (attn[:, :P].cumsum(-1) - 1).clamp(min=0)

    traces = [Trace(machine=TagMachine(p.layout.n_chunks)) for p in prompts]

    # prefill. no masking during the prompt, same as the paper.
    # if every row is the same prompt (sampling n answers to one question, which is what
    # rejection sampling and GRPO do) we prefill once and copy the cache. that's B times less
    # prefill work and no padding
    if B > 1 and all(p.input_ids == prompts[0].input_ids for p in prompts):
        out = model(input_ids=ids[:1], position_ids=pos[:1], use_cache=True, logits_to_keep=1)
        out.cache.batch_repeat_interleave(B)
        cache = out.cache
        logits = out.logits[:, -1].expand(B, -1)
    else:
        out = model(input_ids=ids, attention_mask=attn[:, :P], position_ids=pos, use_cache=True, logits_to_keep=1)
        cache = out.cache
        logits = out.logits[:, -1]

    for step in range(max_new_tokens):
        tok, lp = _sample(logits, temperature, top_p, gen)
        if force is not None:
            tok = torch.tensor([f[step] if step < len(f) else next(iter(eos)) for f in force], device=device)
            lp = torch.log_softmax(logits.float(), -1).gather(1, tok[:, None]).squeeze(1)
        cur = P + step  # column the new token goes into
        for b, t in enumerate(traces):
            if t.finished:
                continue
            tid = int(tok[b])
            if tid in eos:
                t.finished = True
                t.eos_id, t.eos_logprob = tid, float(lp[b])
                continue
            text = tokenizer.decode([tid])
            t.token_ids.append(tid)
            t.token_texts.append(text)
            t.logprobs.append(float(lp[b]))
            t.machine.feed(text)
        if all(t.finished for t in traces) or step == max_new_tokens - 1:
            break

        # new token becomes visible to itself and everything after
        attn[:, cur] = 1
        # finished rows keep generating junk, but we ignore it. keep their mask simple
        step_mask = attn[:, :cur + 1].clone()
        if masked:
            for b, (t, p) in enumerate(zip(traces, prompts)):
                if t.finished or t.machine.mode == "global":
                    continue
                vis = visible_positions(p.layout, t.machine.mode, t.machine.focus_ids, lens[b])
                # prompt positions live at columns offsets[b] .. P-1
                step_mask[b, offsets[b]:P] = vis.to(device).long()
        tok_in = tok.clone()
        for b, t in enumerate(traces):
            if t.finished:
                tok_in[b] = pad_id
        position = torch.tensor([[lens[b] + step] for b in range(B)], device=device)
        out = model(input_ids=tok_in[:, None], attention_mask=step_mask, position_ids=position,
                    cache=cache, use_cache=True)
        cache = out.cache
        logits = out.logits[:, -1]

    for t in traces:
        # decode the whole thing at once. per-token decode can mangle characters that are
        # split across tokens (emoji, accents), which is fine for spotting tags but not for text
        t.text = tokenizer.decode(t.token_ids)
    return traces
