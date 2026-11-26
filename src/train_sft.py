"""LoRA SFT on DA traces, trained under the same mask we evaluate with.

Input: a jsonl where every row has the example fields (context, question, ...) plus
`target`, the trace we want the model to produce. That comes from either
src.synth (warm-up), or src.sample_traces (rejection sampling / templated fallback).

Each example becomes:
    input_ids      = DA prompt + target + <|im_end|>
    labels         = -100 on the prompt, target ids on the completion (we only learn the answer part)
    attention_mask = the full L x L DA mask (masking.training_mask). With masked: false it's
                     just normal causal attention, which is the ablation

We use the plain HF Trainer rather than TRL's SFTTrainer because SFTTrainer builds its own
attention masks and would quietly throw ours away.

    python -m src.train_sft --config configs/sft_synth_0.6b.yaml
"""

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from src.data import read_jsonl
from src.paths import expand
from src.generate import build_prompt, git_commit
from src.masking import training_mask
from src.parse import replay

DEFAULTS = {
    "model": "Qwen/Qwen3-0.6B",
    "data": "data/synth_train.jsonl",
    "out_dir": "checkpoints/sft",
    "masked": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lr": 2e-4,
    "epochs": 1,
    "grad_accum": 8,
    "max_len": 20000,
    "max_examples": None,
    "warmup_steps": 10,
    "save_steps": 100,
    "logging_steps": 5,
    "seed": 0,
    "gradient_checkpointing": True,
    "wandb": False,
}


class DATraceDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, masked, max_len):
        self.items = []
        skipped = 0
        for ex in rows:
            p, _ = build_prompt(tok, ex, "da")
            comp = tok(ex["target"], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
            if p.layout.n_prompt + len(comp) > max_len:
                skipped += 1
                continue
            self.items.append((p, comp))
        self.tok, self.masked = tok, masked
        if skipped:
            print(f"skipped {skipped} examples longer than max_len={max_len}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, comp = self.items[i]
        ids = p.input_ids + comp
        labels = [-100] * len(p.input_ids) + comp
        item = {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}
        if self.masked:
            m = replay([self.tok.decode([t]) for t in comp], p.layout.n_chunks)
            # (1, L, L) -> the collator adds the batch dim. True = may attend
            item["attention_mask"] = training_mask(p.layout, m.modes, (m.mode, m.focus_ids))[None]
        return item


def collate(batch):
    # batch size is 1 on purpose (every example has its own LxL mask and lengths vary a lot),
    # so collating is just adding a batch dim
    assert len(batch) == 1, "use per_device_train_batch_size=1 and grad accumulation"
    return {k: v[None] for k, v in batch[0].items()}


def main(argv=None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--max-examples", dest="max_examples", type=int)
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--grad-accum", dest="grad_accum", type=int)
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    cfg = {k: expand(v) for k, v in cfg.items()}

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = read_jsonl(cfg["data"])
    random.Random(cfg["seed"]).shuffle(rows)
    if cfg["max_examples"]:
        rows = rows[:cfg["max_examples"]]
    ds = DATraceDataset(rows, tok, cfg["masked"], cfg["max_len"])
    print(f"{len(ds)} training examples, masked={cfg['masked']}")

    cuda = torch.cuda.is_available()
    # bf16 weights on gpu and on mac (fp32 on an 8GB mac swaps). plain cpu gets fp32
    on_gpu = cuda or torch.backends.mps.is_available()
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16 if on_gpu else torch.float32,
                                                 attn_implementation="sdpa")
    if cfg["gradient_checkpointing"]:
        # recompute activations in the backward pass instead of storing them. slower, but it's
        # the difference between fitting a 16K sequence in memory or not
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules="all-linear", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**cfg, "git_commit": git_commit()}, indent=2))

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps") or -1,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=cuda,
        report_to=["wandb"] if cfg["wandb"] else [],
        run_name=out.name,
        remove_unused_columns=False,
        seed=cfg["seed"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    # resume automatically if there's a checkpoint from a session that died
    has_ckpt = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    print(f"saved adapter to {out / 'final'}")


if __name__ == "__main__":
    main()
