"""Run a model over a dataset in one or more arms and save every trace.

Arms:
  vanilla    the paper's vanilla prompt (whole document inline, no tags)
  da_nomask  DA prompt, but nothing is hidden. tells us how well the model *follows the
             protocol*, separately from whether it can cope with the masking
  da         DA prompt with real masking. the actual number

    python -m src.generate --config configs/baseline_qwen3_4b.yaml
    python -m src.generate --config configs/smoke.yaml --limit 3     # quick local check

Output goes to <out_dir>/<run_name>/:
  traces.jsonl   one line per (example, arm, seed). appended as we go, so a crashed Colab
                 session can just be restarted with the same command and it skips what's done
  meta.json      config, git commit, model, start time
  metrics.csv / summary.csv   written by src.report at the end
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

from src.chunking import chunks_containing
from src.data import read_jsonl
from src.decode import generate
from src.prompts import build_da, build_vanilla
from src.synth import chunks_from_example

DEFAULTS = {
    "model": "Qwen/Qwen3-0.6B",
    "adapter": None,          # path to a LoRA adapter, for evaluating fine-tuned models
    "data": "data/niah_test.jsonl",
    "arms": ["vanilla", "da_nomask", "da"],
    "seeds": [0],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "batch_size": 4,
    "limit": None,
    "buckets": None,          # e.g. ["4k"] to only run one length
    "out_dir": "runs",
    "run_name": None,
    "dtype": "auto",
    "wandb": False,
}


def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if not cfg["run_name"]:
        cfg["run_name"] = Path(path).stem if path else "run"
    return cfg


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # fp32 on an 8GB mac swaps itself to death (a 3 example smoke run took 10+ minutes),
    # so bf16 everywhere except plain cpu
    dtype = {"auto": torch.float32 if device == "cpu" else torch.bfloat16,
             "bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[cfg["dtype"]]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    # sdpa handles arbitrary masks. flash-attention 2 would NOT respect our holes in the mask
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype, attn_implementation="sdpa")
    if cfg.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter"])
        model = model.merge_and_unload()
    return model.to(device).eval(), tok


def build_prompt(tok, ex, arm):
    if arm == "vanilla":
        return build_vanilla(tok, ex["context"], ex["question"]), ex.get("needle_chunks")
    chunks = chunks_from_example(ex) if "chunk_texts" in ex else None
    p = build_da(tok, ex["context"], ex["question"], chunks=chunks)
    needle = ex.get("needle_chunks") or chunks_containing(p.chunks, ex["answer"])
    return p, needle


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--data")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--run-name", dest="run_name")
    ap.add_argument("--arms", nargs="+")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {k: v for k, v in vars(args).items() if k != "config"})

    out = Path(cfg["out_dir"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "traces.jsonl"
    done = set()
    if trace_path.exists():
        for r in read_jsonl(trace_path):
            done.add((r["id"], r["arm"], r["seed"]))
        print(f"resuming, {len(done)} traces already done")
    meta = {"config": cfg, "git_commit": git_commit(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    examples = read_jsonl(cfg["data"])
    if cfg["buckets"]:
        examples = [e for e in examples if e["bucket"] in cfg["buckets"]]
    if cfg["limit"]:
        examples = examples[:cfg["limit"]]

    device = pick_device()
    model, tok = load_model(cfg, device)
    print(f"{cfg['model']} on {device}, {len(examples)} examples, arms {cfg['arms']}, seeds {cfg['seeds']}")

    if cfg["wandb"]:
        import wandb
        wandb.init(project="learning-to-focus", name=cfg["run_name"], config=cfg, job_type="eval")

    t0 = time.time()
    with open(trace_path, "a") as f:
        for seed in cfg["seeds"]:
            for arm in cfg["arms"]:
                todo = [e for e in examples if (e["id"], arm, seed) not in done]
                # similar lengths together = less padding waste
                todo.sort(key=lambda e: len(e["context"]))
                for i in range(0, len(todo), cfg["batch_size"]):
                    batch = todo[i:i + cfg["batch_size"]]
                    built = [build_prompt(tok, e, arm) for e in batch]
                    traces = generate(
                        model, tok, [p for p, _ in built], masked=(arm == "da"),
                        max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
                        top_p=cfg["top_p"], seed=seed * 100003 + i,
                    )
                    for ex, (p, needle), tr in zip(batch, built, traces):
                        f.write(json.dumps({
                            "id": ex["id"], "task": ex["task"], "bucket": ex["bucket"], "arm": arm, "seed": seed,
                            "gold": ex["answer"], "needle_chunks": needle, "layout": p.layout.to_dict(),
                            "text": tr.text, "token_ids": tr.token_ids, "finished": tr.finished,
                        }) + "\n")
                    f.flush()
                    n = i + len(batch)
                    print(f"  seed {seed} {arm}: {n}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)

    from src.report import report
    summary = report(out, tok)
    if cfg["wandb"]:
        import wandb
        wandb.log({"summary": wandb.Table(dataframe=summary)})
        wandb.finish()


if __name__ == "__main__":
    main()
