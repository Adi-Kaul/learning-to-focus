"""Tiny fake documents for the warm-up experiment.

Each document is 3-5 magic chunks of ~100-200 tokens of boring filler. One chunk hides the
fact we ask about ("The code for blue is 7731."). Some other chunks hide distractor facts
with different keys, so the model actually has to pick the right chunk instead of
focusing on whichever one has a number in it.

Also builds an "ideal" DA trace for each example, from a few templates. That's the SFT
target for the warm-up, and the same helper works for RULER questions.

    python -m src.synth --n-train 500 --n-test 200
"""

import argparse
import random
import re
from pathlib import Path

from src.chunking import Chunk

FILLER = [
    "The meeting was moved to the afternoon because the projector was broken.",
    "Most of the shelves in the back room were empty by the end of the week.",
    "Someone left a half finished crossword on the table near the window.",
    "The train was late again, which surprised nobody.",
    "There were three kinds of tea in the cupboard and none of them were good.",
    "The report had been printed twice by mistake, so there were extra copies.",
    "A delivery truck blocked the street for almost an hour.",
    "The plants on the balcony needed water but everyone forgot.",
    "Nobody could remember who had ordered the large box of pens.",
    "The wifi password was written on a sticky note that had fallen behind the desk.",
    "It rained most of the morning and then cleared up around noon.",
    "The new chairs were more comfortable than the old ones, mostly.",
    "A long email thread argued about the font used on the posters.",
    "The coffee machine made a strange noise but still worked.",
    "Two people showed up wearing almost the same sweater.",
    "The elevator was out of service, so everyone took the stairs.",
    "Somebody suggested a group lunch, and then nobody organized it.",
    "The calendar on the wall was still showing last month.",
    "A stack of old magazines sat untouched in the waiting area.",
    "The heating came on too early and the room got very warm.",
    "Several boxes were labeled fragile even though they held paper.",
    "The cleaning crew came in later than usual on Thursday.",
    "A map of the city was pinned up next to the fire exit.",
    "The parking lot was half empty because of the holiday.",
    "Someone kept humming the same song all afternoon.",
    "The library returned a book that nobody remembered borrowing.",
    "The printer ran out of toner right before the big deadline.",
    "A small dog waited patiently outside the bakery.",
    "The windows were cleaned on Monday and were dirty again by Friday.",
    "A list of phone numbers was taped inside the supply closet.",
]
KEYS = ["blue", "red", "green", "orange", "purple", "silver", "gold", "black", "white", "yellow",
        "falcon", "otter", "maple", "cedar", "lynx", "heron", "copper", "ivory", "coral", "slate",
        "tiger", "pine", "willow", "amber", "jade", "onyx", "raven", "bison", "lotus", "cobalt"]
QUESTION = "What is the code for {key}?"
FACT = "The code for {key} is {value}."


def make_example(rng, idx, tokenizer=None):
    n_chunks = rng.randint(3, 5)
    keys = rng.sample(KEYS, k=1 + rng.randint(0, min(2, n_chunks - 1)))
    target_key, distractors = keys[0], keys[1:]
    values = {k: str(rng.randint(1000, 9999)) for k in keys}
    target_chunk = rng.randint(1, n_chunks)
    other_chunks = [c for c in range(1, n_chunks + 1) if c != target_chunk]
    fact_chunk = {target_key: target_chunk}
    for k, c in zip(distractors, rng.sample(other_chunks, len(distractors))):
        fact_chunk[k] = c

    texts = []
    for cid in range(1, n_chunks + 1):
        # ~6-11 sentences is roughly 100-200 tokens
        sents = rng.sample(FILLER, rng.randint(6, 11))
        for k, c in fact_chunk.items():
            if c == cid:
                sents.insert(rng.randint(0, len(sents)), FACT.format(key=k, value=values[k]))
        texts.append(" ".join(sents) + "\n\n")

    chunks, pos = [], 0
    for cid, t in enumerate(texts, start=1):
        n = len(tokenizer(t, add_special_tokens=False)["input_ids"]) if tokenizer else 0
        chunks.append(Chunk(cid, t, n, pos, pos + len(t)))
        pos += len(t)

    return {
        "id": f"synth/{idx}",
        "task": "synth",
        "bucket": f"{n_chunks}chunks",
        "context": "".join(texts),
        "chunk_texts": texts,   # keep the exact chunking, we don't want the real chunker merging these
        "question": QUESTION.format(key=target_key),
        "answer": values[target_key],
        "needle_chunks": [target_chunk],
        "key": target_key,
    }


# a few shapes so the model doesn't just memorise one exact string.
# all of them follow the paper's advice: at least one focus, end with a local that commits
TEMPLATES = [
    ('<global>The question asks for the {what} for {key}. Magic chunk {k} contains it.</global>\n'
     '<focus magic_chunks="{k}">{value}</focus>\n'
     '<local>The {what} for {key} is {value}.</local>\n'
     '<answer>{value}</answer>'),
    ('<local>I need to find the {what} for {key}.</local>\n'
     '<global>Magic chunk {k} mentions the {what} for {key}.</global>\n'
     '<focus magic_chunks="{k}">{value}</focus>\n'
     '<local>So the answer is {value}.</local>\n'
     '<answer>{value}</answer>'),
    ('<global>Magic chunk {k} is where {key} comes up.</global>\n'
     '<focus magic_chunks="{k}">{value}</focus>\n'
     '<local>Extracted {value} for {key}. That is the answer.</local>\n'
     '<answer>{value}</answer>'),
]


def ideal_trace(key, value, chunk_id, rng, what="code"):
    return rng.choice(TEMPLATES).format(key=key, value=value, k=chunk_id, what=what)


RULER_Q = re.compile(r"special magic (\w+) for (.+?) mentioned")


def ruler_key(question):
    """'What is the special magic number for wandering-age mentioned...' -> ('special magic number', 'wandering-age')"""
    m = RULER_Q.search(question)
    if not m:
        return None, None
    return f"special magic {m.group(1)}", m.group(2)


def chunks_from_example(ex):
    """Rebuild Chunk objects for a synth example (they come pre-chunked)."""
    chunks, pos = [], 0
    for cid, t in enumerate(ex["chunk_texts"], start=1):
        chunks.append(Chunk(cid, t, 0, pos, pos + len(t)))
        pos += len(t)
    return chunks


def main():
    from src.data import DATA, write_jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [make_example(rng, i) for i in range(args.n_train + args.n_test)]
    test, train = rows[:args.n_test], rows[args.n_test:]
    # same leak rule as RULER: no test answer value can show up in train
    test_pairs = {(e["key"], e["answer"]) for e in test}
    train = [e for e in train if (e["key"], e["answer"]) not in test_pairs]
    for e in train:
        e["target"] = ideal_trace(e["key"], e["answer"], e["needle_chunks"][0], rng)
    write_jsonl(train, Path(DATA) / "synth_train.jsonl")
    write_jsonl(test, Path(DATA) / "synth_test.jsonl")
    print(f"synth: {len(train)} train, {len(test)} test")


if __name__ == "__main__":
    main()
