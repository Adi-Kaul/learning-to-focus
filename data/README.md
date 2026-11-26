Generated data lives here and is gitignored, except the split file.

    python -m src.data --test-per-bucket 20    # -> niah_train.jsonl, niah_test.jsonl
    python -m src.synth                        # -> synth_train.jsonl, synth_test.jsonl (warm-up)

`splits/niah_test_ids.txt` is committed so the test set can't silently change.
297 rows: 20 per (task, length) except single_1/8k (19) and single_2/8k (18), which
came up short because of how needles are shared across rows (see src/data.py).
