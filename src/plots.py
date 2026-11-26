"""Charts for the write-up. Reads metrics.csv from one or more runs, writes PNGs.

    python -m src.plots runs/baseline_qwen3_4b_4k --out blog/figures
    python -m src.plots runs/baseline_qwen3_4b runs/eval_sft_qwen3_4b runs/eval_grpo_qwen3_4b --out blog/figures

Every "arm" gets a label like "vanilla", "DA (no mask)", "DA", and for fine-tuned runs
"DA + SFT", "DA + GRPO" (taken from the run name). Colors are fixed per label so the same
thing is the same color in every chart.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.failures import CATEGORIES, categorize  # noqa: E402
from src.data import read_jsonl  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed color per thing. vanilla is the neutral reference, the DA variants get the
# validated categorical slots in order (blue, orange, aqua), then violet/red for later stages
COLORS = {
    "vanilla": "#8a8984",
    "DA (no mask)": "#2a78d6",
    "DA": "#eb6834",
    "DA + SFT": "#1baf7a",
    "DA + GRPO": "#6250d6",
}
ORDER = list(COLORS)
BUCKETS = ["4k", "8k", "16k"]


def label_for(run_name, arm):
    if arm == "vanilla":
        return "vanilla"
    if arm == "da_nomask":
        return "DA (no mask)"
    if "grpo" in run_name:
        return "DA + GRPO"
    if "sft" in run_name:
        return "DA + SFT"
    return "DA"


def load(runs):
    frames = []
    for r in runs:
        df = pd.read_csv(Path(r) / "metrics.csv")
        df["label"] = [label_for(Path(r).name, a) for a in df["arm"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def accuracy_by_length(df, out):
    labels = [lab for lab in ORDER if lab in set(df["label"])]
    buckets = [b for b in BUCKETS if b in set(df["bucket"])]
    acc = df.groupby(["label", "bucket"])["correct"].mean()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        xs = [j + (i - (len(labels) - 1) / 2) * w for j in range(len(buckets))]
        ys = [acc.get((lab, b), float("nan")) * 100 for b in buckets]
        ax.bar(xs, ys, width=w * 0.92, color=COLORS[lab], label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(buckets)), [f"~{b} tokens" for b in buckets])
    ax.set_ylim(0, 105)
    style(ax, "Accuracy by context length", "accuracy (%)")
    ax.legend(frameon=False, fontsize=9, ncol=len(labels), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_length.png", dpi=200)
    plt.close(fig)


def accuracy_vs_reading(df, out):
    """The headline chart: how right is it vs how much did it read. Up and to the left is good."""
    df = df.copy()
    # the no-mask arm really did read everything, so plot what it actually cost, not the
    # simulated "if it had been masked" number (that would flatter it)
    nomask = df["label"] == "DA (no mask)"
    df.loc[nomask, "attended"] = df.loc[nomask, "dense_attended"]
    g = df.groupby("label").agg(acc=("correct", "mean"), att=("attended", "sum"), n=("correct", "size"))
    van = g.loc["vanilla", "att"] if "vanilla" in g.index else g["att"].max()
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    for lab in [x for x in ORDER if x in g.index]:
        rel = g.loc[lab, "att"] / van * 100
        acc = g.loc[lab, "acc"] * 100
        ax.scatter(rel, acc, s=70, color=COLORS[lab], edgecolor=SURFACE, linewidth=2, zorder=3, label=lab)
        ax.annotate(lab, (rel, acc), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(0, max(115, (g["att"] / van * 100).max() * 1.15))
    ax.set_ylim(0, 105)
    ax.set_xlabel("attended tokens, % of vanilla", color=INK2, fontsize=10)
    style(ax, "Accuracy vs. how much the model reads", "accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_reading.png", dpi=200)
    plt.close(fig)


def failure_breakdown(runs, out):
    counts = {c: 0 for c in CATEGORIES}
    for r in runs:
        for row in read_jsonl(Path(r) / "traces.jsonl"):
            if row["arm"] == "da" and "sft" not in Path(r).name and "grpo" not in Path(r).name:
                c = categorize(row)
                if c:
                    counts[c] += 1
    total = sum(counts.values())
    if not total:
        return
    cats = [c for c in CATEGORIES if counts[c]][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(cats) + 1.2), facecolor=SURFACE)
    ax.barh(range(len(cats)), [counts[c] for c in cats], color=COLORS["DA"], height=0.6)
    for i, c in enumerate(cats):
        ax.text(counts[c] + total * 0.01, i, f"{counts[c]} ({counts[c] / total:.0%})", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
    style(ax, f"Why masked DA answers were wrong (n={total})", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out / "failure_breakdown.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="blog/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs)
    accuracy_by_length(df, out)
    accuracy_vs_reading(df, out)
    failure_breakdown(args.runs, out)
    print(f"wrote charts to {out}")


if __name__ == "__main__":
    main()
