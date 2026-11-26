"""GRPO with TRL, where both the rollouts and the training forward pass use the DA mask.

Two things TRL doesn't do out of the box, and why we need them:

1. Rollouts. TRL's built-in generation (HF generate or vLLM) can't change the attention mask
   mid-generation. So we hand it a `rollout_func` that calls our own masked decode loop.
   Otherwise the model would be rewarded for answers it could only give because it could
   still see the document in <local> mode.

2. Log-probs. GRPO pushes up the probability of good completions, which needs a forward
   pass over prompt + completion. If that pass used normal attention, we'd be training a
   slightly different model from the one that generated the rollouts (and the one we
   evaluate). So MaskedGRPOTrainer overrides the log-prob function to use the same L x L
   mask as masking.training_mask.

Rewards are split into three functions (correct / format / efficiency) so TRL logs each
one separately. Watching them move separately is the main way to catch reward hacking.

    python -m src.train_grpo --config configs/grpo_qwen3_4b.yaml
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from src.data import read_jsonl
from src.decode import generate
from src.paths import expand
from src.generate import build_prompt, git_commit
from src.masking import training_mask
from src.parse import replay
from src.prompts import Layout
from src.rewards import reward_parts

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

DEFAULTS = {
    "model": "Qwen/Qwen3-0.6B",
    "adapter": None,                 # start from the SFT adapter (strongly recommended)
    "data": "data/grpo_train.jsonl",  # needs vanilla_attended, see sample_traces --vanilla-ref
    "out_dir": "checkpoints/grpo",
    "masked": True,
    "lam": 0.3,
    "num_generations": 8,
    "prompts_per_step": 4,
    "max_completion_length": 512,
    "temperature": 0.8,
    "lr": 2e-5,
    "beta": 0.0,
    "max_steps": 200,
    "save_steps": 25,
    "logging_steps": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_examples": None,
    "seed": 0,
    "wandb": False,
}


def make_trainer_class():
    # defined inside a function so importing this module doesn't drag in trl
    from trl import GRPOTrainer

    class _MaskedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, da_masked=True, prompt_layouts=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.da_masked = da_masked
            # tuple(prompt token ids) -> Layout. filled in by the rollout function
            self.prompt_layouts = prompt_layouts if prompt_layouts is not None else {}

        def _row_mask(self, ids, am, logits_to_keep):
            """Build the (L, L) DA mask + position ids for one padded row of prompt+completion."""
            L = ids.size(0)
            P_pad = L - logits_to_keep            # prompt part, left padded
            pad = int((am[:P_pad] == 0).sum())    # leading pads
            prompt = tuple(ids[pad:P_pad].tolist())
            layout = self.prompt_layouts[prompt]
            T = int(am[P_pad:].sum())             # real completion length (rest is right padding)
            comp = ids[P_pad:P_pad + T].tolist()
            texts = [self.processing_class.decode([t]) for t in comp]
            m = replay(texts, layout.n_chunks)
            inner = training_mask(layout, m.modes, (m.mode, m.focus_ids))
            full = torch.eye(L, dtype=torch.bool)  # pads only see themselves, so no row is all -inf
            n = layout.n_prompt + T
            full[pad:pad + n, pad:pad + n] = inner
            pos = torch.zeros(L, dtype=torch.long)
            pos[pad:pad + n] = torch.arange(n)
            return full, pos

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                               batch_size=None, compute_entropy=False, **kwargs):
            if not self.da_masked:
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size,
                    compute_entropy=compute_entropy, **kwargs)
            logps, ents = [], []
            # one row at a time: every row has its own mask, and memory is the bottleneck anyway
            for b in range(input_ids.size(0)):
                ids, am = input_ids[b], attention_mask[b]
                mask, pos = self._row_mask(ids, am, logits_to_keep)
                labels = ids.masked_fill(am == 0, -100)[None].clone()
                labels[:, :-logits_to_keep] = -100
                with self.accelerator.autocast():
                    out = model(input_ids=ids[None], attention_mask=mask[None, None].to(ids.device),
                                position_ids=pos[None].to(ids.device), labels=labels, fused_lm_head=True)
                logps.append(out.log_probs[:, -logits_to_keep:])
                if compute_entropy:
                    ents.append(out.entropy[:, -logits_to_keep:])
            return torch.cat(logps), (torch.cat(ents) if compute_entropy else None), None

    return _MaskedGRPOTrainer


def make_rollout_func(examples_by_id, tok, cfg, prompt_layouts):
    """TRL calls this with the batch's prompts (our dataset 'prompt' column = example id),
    each repeated num_generations times in a row."""
    cache = {}

    def rollout(prompts, trainer):
        model = trainer.model
        was_training = model.training
        model.eval()
        prompt_ids, completion_ids, logprobs, layouts = [], [], [], []
        i = 0
        while i < len(prompts):
            j = i
            while j < len(prompts) and prompts[j] == prompts[i]:
                j += 1
            ex = examples_by_id[prompts[i]]
            if ex["id"] not in cache:
                p, _ = build_prompt(tok, ex, "da")
                cache[ex["id"]] = p
                prompt_layouts[tuple(p.input_ids)] = p.layout
            p = cache[ex["id"]]
            traces = generate(model, tok, [p] * (j - i), masked=cfg["masked"],
                              max_new_tokens=cfg["max_completion_length"], temperature=cfg["temperature"],
                              seed=trainer.state.global_step * 1000 + i)
            for tr in traces:
                ids, lps = list(tr.token_ids), list(tr.logprobs)
                if tr.finished:  # include EOS so the model also learns when to stop
                    ids.append(tr.eos_id)
                    lps.append(tr.eos_logprob)
                prompt_ids.append(list(p.input_ids))
                completion_ids.append(ids)
                logprobs.append(lps)
                layouts.append(p.layout.to_dict())
            i = j
        if was_training:
            model.train()
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs,
                "layout": layouts}

    return rollout


def make_reward_funcs(tok, lam):
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

    def parts(completion_ids, layout, answer, vanilla_attended):
        out = []
        for ids, lay, gold, ref in zip(completion_ids, layout, answer, vanilla_attended):
            ids = [t for t in ids if t not in eos]
            texts = [tok.decode([t]) for t in ids]
            out.append(reward_parts(tok.decode(ids), texts, Layout.from_dict(lay), gold, reference=ref, lam=lam))
        return out

    # TRL calls each reward function with the completions plus every dataset column / extra field
    def correct(completions, completion_ids, layout, answer, vanilla_attended, **kw):
        return [r["r_correct"] for r in parts(completion_ids, layout, answer, vanilla_attended)]

    def well_formed(completions, completion_ids, layout, answer, vanilla_attended, **kw):
        return [r["r_format"] for r in parts(completion_ids, layout, answer, vanilla_attended)]

    def efficiency(completions, completion_ids, layout, answer, vanilla_attended, **kw):
        return [lam * r["r_correct"] * r["r_efficiency"] for r in parts(completion_ids, layout, answer, vanilla_attended)]

    return [correct, well_formed, efficiency]


def main(argv=None):
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--max-examples", dest="max_examples", type=int)
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--lam", type=float)
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    cfg = {k: expand(v) for k, v in cfg.items()}

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = read_jsonl(cfg["data"])
    if cfg["max_examples"]:
        rows = rows[:cfg["max_examples"]]
    missing = sum("vanilla_attended" not in r for r in rows)
    if missing:
        print(f"WARNING: {missing} rows have no vanilla_attended, efficiency falls back to the same-trace "
              "ratio which doesn't punish long outputs. run sample_traces --vanilla-ref first")
    examples_by_id = {r["id"]: r for r in rows}
    ds = Dataset.from_list([{"prompt": r["id"], "answer": r["answer"],
                             "vanilla_attended": r.get("vanilla_attended")} for r in rows])

    cuda = torch.cuda.is_available()
    # bf16 weights on gpu and on mac (fp32 on an 8GB mac swaps). plain cpu gets fp32
    on_gpu = cuda or torch.backends.mps.is_available()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if on_gpu else torch.float32,
                                                 attn_implementation="sdpa")
    peft_config = None
    if cfg["adapter"]:
        # keep training the SFT adapter rather than stacking a second one on top
        model = PeftModel.from_pretrained(model, cfg["adapter"], is_trainable=True)
    else:
        peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], target_modules="all-linear",
                                 task_type="CAUSAL_LM")

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**cfg, "git_commit": git_commit()}, indent=2))

    n = cfg["num_generations"]
    gargs = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=n,           # one prompt's worth of completions per micro batch
        gradient_accumulation_steps=cfg["prompts_per_step"],
        num_generations=n,
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        learning_rate=cfg["lr"],
        beta=cfg["beta"],
        max_steps=cfg["max_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        logging_steps=cfg["logging_steps"],
        bf16=cuda,
        gradient_checkpointing=True,
        report_to=["wandb"] if cfg["wandb"] else [],
        run_name=out.name,
        seed=cfg["seed"],
        log_completions=True,
        num_completions_to_print=2,
        remove_unused_columns=False,
    )
    prompt_layouts = {}
    Trainer = make_trainer_class()
    trainer = Trainer(
        model=model,
        reward_funcs=make_reward_funcs(tok, cfg["lam"]),
        args=gargs,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        rollout_func=make_rollout_func(examples_by_id, tok, cfg, prompt_layouts),
        da_masked=cfg["masked"],
        prompt_layouts=prompt_layouts,
    )
    has_ckpt = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(str(out / "final"))
    print(f"saved to {out / 'final'}")


if __name__ == "__main__":
    main()
