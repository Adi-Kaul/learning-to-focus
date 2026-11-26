"""RULER needle-in-a-haystack data, and the train/test split.

Instead of generating RULER ourselves we use the exact rows the paper's authors evaluated on:
the dac-research/longbench_synthetic_v4_1 dataset on the HF hub. Each of the 5 NIAH tasks
has 1500 rows: 500 at ~4K tokens, 500 at ~8K and 500 at ~16K (multikey runs a bit longer,
its "16K" bucket is closer to 20K).

  niah_single_1    number needle in a repetitive "the grass is green" haystack
  niah_single_2    number needle in Paul Graham essays
  niah_single_3    uuid needle in Paul Graham essays
  niah_multikey_2  number needle surrounded by lots of other "magic number for X" needles
  niah_multikey_3  uuid -> uuid, also surrounded by distractors

Split: careful here. RULER reuses needles. In the single_* tasks the same "magic number for
solid-few is 8090293" shows up at 4K, 8K and 16K, and single_1/single_2 share needles too.
A plain random split by row would put test answers in the training set (the leak check
below caught exactly that the first time). So rows are grouped by needle (same answer or
same question = same group, merged transitively) and whole groups go to one side.

We pick groups for test until every (task, length bucket) has N rows. The test ids get
written to data/splits/ and committed, so the split can't drift later.

Usage:
    python -m src.data --test-per-bucket 20
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO = "dac-research/longbench_synthetic_v4_1"
TASKS = ["ruler_niah_single_1", "ruler_niah_single_2", "ruler_niah_single_3",
         "ruler_niah_multikey_2", "ruler_niah_multikey_3"]
BUCKETS = ["4k", "8k", "16k"]
DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def load_task(task):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, f"{task}/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["context_tokens"] = df["context_tokens"].astype(int)
    # three length groups of 500. rank instead of hard thresholds since multikey runs longer
    order = df["context_tokens"].rank(method="first").astype(int) - 1
    df["bucket"] = [BUCKETS[min(i * 3 // len(df), 2)] for i in order]
    df["task"] = task
    df["id"] = [f"{task}/{i}" for i in df["source_index"]]
    return df


def to_examples(df):
    return [
        {"id": r.id, "task": r.task, "bucket": r.bucket, "context": r.context, "question": r.question,
         "answer": r.reference_answer, "context_tokens": int(r.context_tokens)}
        for r in df.itertuples()
    ]


def needle_groups(rows):
    """Union-find over rows: two rows are in the same group if they share an answer or a
    question. Returns a list of groups (lists of rows)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        for key in (("a", r["answer"]), ("q", r["question"])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    return list(groups.values())


def make_split(test_per_bucket=20, seed=SEED):
    rows = []
    for task in TASKS:
        rows += to_examples(load_task(task))
    groups = needle_groups(rows)
    # sort first so the shuffle doesn't depend on dict/hash ordering
    groups.sort(key=lambda g: min(r["id"] for r in g))
    random.Random(seed).shuffle(groups)

    count = {(t, b): 0 for t in TASKS for b in BUCKETS}
    train, test = [], []
    for g in groups:
        cells = [(r["task"], r["bucket"]) for r in g]
        # a group goes to test only if it fits in every cell it touches
        fits = all(count[c] + cells.count(c) <= test_per_bucket for c in set(cells))
        if fits and any(count[c] < test_per_bucket for c in cells):
            for c in cells:
                count[c] += 1
            test += g
        else:
            train += g
    short = {c: n for c, n in count.items() if n < test_per_bucket}
    if short:
        print("warning: some cells came up short:", short)
    check_no_leak(train, test)
    return train, test


def check_no_leak(train, test):
    test_ids = {e["id"] for e in test}
    assert not test_ids & {e["id"] for e in train}, "same row in train and test"
    train_q = {e["question"] for e in train}
    assert not [e for e in test if e["question"] in train_q], "test question also in train"
    # the needle value is what we grade on. if a test answer shows up as a train answer,
    # training could just teach the model that string
    train_answers = {e["answer"] for e in train}
    clash = [e["id"] for e in test if e["answer"] in train_answers]
    assert not clash, f"test answers also appear in train: {clash[:5]}"


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-per-bucket", type=int, default=20)
    args = ap.parse_args()
    train, test = make_split(args.test_per_bucket)
    write_jsonl(train, DATA / "niah_train.jsonl")
    write_jsonl(test, DATA / "niah_test.jsonl")
    (DATA / "splits").mkdir(parents=True, exist_ok=True)
    (DATA / "splits" / "niah_test_ids.txt").write_text("\n".join(e["id"] for e in test) + "\n")
    print(f"train {len(train)}  test {len(test)}  -> {DATA}")


if __name__ == "__main__":
    main()
