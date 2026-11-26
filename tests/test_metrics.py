import math

from src.metrics import attended_tokens, dense_attended_tokens, exact_match, score_trace, summarize, visible_count
from src.parse import FOCUS, GLOBAL, LOCAL
from src.prompts import Layout

# a made up prompt: 1000 tokens, sink 16, three chunks, question starts at 900
L = Layout(n_prompt=1000, question_start=900, chunk_spans=[(100, 400), (400, 700), (700, 890)])


def test_all_global_equals_dense():
    # sanity check #1 from the plan: an all-global trace costs the same as vanilla
    modes = [(GLOBAL, ())] * 50
    assert attended_tokens(L, modes) == dense_attended_tokens(1000, 50)


def test_hand_computed():
    # sanity check #2: small trace we can do on paper
    # step 1 global: seq 1001 -> 1001
    # step 2 focus on chunk 2 (300 tokens): seq 1002 -> 16 + 300 + (1002 - 900) = 418
    # step 3 local: seq 1003 -> 16 + 103 = 119
    modes = [(GLOBAL, ()), (FOCUS, (2,)), (LOCAL, ())]
    assert attended_tokens(L, modes) == 1001 + 418 + 119


def test_duplicate_focus_ids_count_once():
    assert visible_count(L, FOCUS, (1, 1), 1001) == visible_count(L, FOCUS, (1,), 1001)


def test_exact_match():
    assert exact_match(" 8090293 ", "8090293")
    assert exact_match("ABC-def", "abc-DEF")
    assert not exact_match(None, "1")
    assert not exact_match("8090293.", "8090293")  # trailing period is wrong. strict on purpose


def test_score_trace_end_to_end():
    pieces = ["<focus magic_chunks=\"2\">", "777", "</focus>", "<local>", "ok", "</local>", "<answer>", "777", "</answer>"]
    row = score_trace("".join(pieces), pieces, L, gold="777", needle_chunks=[2])
    assert row["correct"] and row["parse_ok"] and row["needle_hit"]
    assert row["focus_success_rate"] == 1.0
    assert row["attended"] < row["dense_attended"]
    assert row["focus_steps"] == 2 and row["local_steps"] == 2 + 2  # local + answer blocks


def test_vanilla_row():
    row = score_trace("<answer>5</answer>", ["<answer>", "5", "</answer>"], Layout(500, 400, []), "5", is_da=False)
    assert row["attended"] == dense_attended_tokens(500, 3)


def test_summarize():
    rows = [
        {"correct": True, "has_answer": True, "decode_steps": 10, "attended": 50, "dense_attended": 100,
         "focus_attempts": 1, "focus_successes": 1, "focus_success_rate": 1.0},
        {"correct": False, "has_answer": True, "decode_steps": 20, "attended": 100, "dense_attended": 100,
         "focus_attempts": 0, "focus_successes": 0, "focus_success_rate": math.nan},
    ]
    s = summarize(rows)
    assert s["correct"] == 0.5
    assert s["focus_success_rate"] == 1.0  # nan skipped
    assert s["attention_ratio"] == 0.75
