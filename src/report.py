"""Score a run's traces and print the tables.

Metrics are always recomputed from traces.jsonl, never stored at generation time, so if we
fix a bug in metrics.py we just rerun this instead of regenerating anything.

    python -m src.report runs/baseline_qwen3_4b
    python -m src.report runs/a runs/b --by arm bucket     # several runs side by side
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import read_jsonl
from src.metrics import score_trace, summarize
from src.prompts import Layout


def score_run(run_dir, tok):
    rows = []
    for r in read_jsonl(Path(run_dir) / "traces.jsonl"):
        texts = [tok.decode([i]) for i in r["token_ids"]]
        layout = Layout.from_dict(r["layout"])
        m = score_trace(r["text"], texts, layout, r["gold"], r.get("needle_chunks"), is_da=r["arm"] != "vanilla")
        m.update({k: r[k] for k in ("id", "task", "bucket", "arm", "seed", "finished")})
        m["run"] = Path(run_dir).name
        rows.append(m)
    return rows


def table(rows, by):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        s = summarize(rs)
        s["ran_out"] = sum(not r["finished"] for r in rs) / len(rs)
        out.append({**dict(zip(by, key)), **s})
    return pd.DataFrame(out)


def report(run_dir, tok=None, by=("arm", "bucket")):
    if tok is None:
        from transformers import AutoTokenizer
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        tok = AutoTokenizer.from_pretrained(meta["config"]["model"])
    rows = score_run(run_dir, tok)
    pd.DataFrame(rows).to_csv(Path(run_dir) / "metrics.csv", index=False)
    summary = table(rows, list(by))
    summary.to_csv(Path(run_dir) / "summary.csv", index=False)
    show = ["correct", "parse_ok", "focus_success_pooled", "needle_hit", "decode_steps", "attention_ratio",
            "attended_per_example", "ran_out", "n"]
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(summary[[*by, *[c for c in show if c in summary]]].to_string(index=False))
        print()
        print(table(rows, ["arm"])[["arm", *[c for c in show if c in summary]]].to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--by", nargs="+", default=["arm", "bucket"])
    args = ap.parse_args()
    for r in args.runs:
        print(f"== {r}")
        report(r, by=tuple(args.by))


if __name__ == "__main__":
    main()
