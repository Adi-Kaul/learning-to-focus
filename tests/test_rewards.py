import pytest

from src.prompts import Layout
from src.rewards import FORMAT_BONUS, efficiency, reward_parts

L = Layout(n_prompt=1000, question_start=900, chunk_spans=[(100, 400), (400, 700), (700, 890)])


def split(s):
    # fake "tokens": split right after every '>' so tags end a token, like real tokenizers mostly do
    out, cur = [], ""
    for ch in s:
        cur += ch
        if ch == ">":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


GOOD = '<focus magic_chunks="2">777</focus><local>it is 777</local><answer>777</answer>'


def test_good_trace():
    r = reward_parts(GOOD, split(GOOD), L, "777", reference=100_000)
    assert r["r_correct"] == 1 and r["r_format"] == FORMAT_BONUS
    assert 0 < r["r_efficiency"] < 1
    assert r["reward"] == pytest.approx(1 + FORMAT_BONUS + 0.3 * r["r_efficiency"])


def test_wrong_answer_gets_no_efficiency():
    t = GOOD.replace("<answer>777", "<answer>778")
    r = reward_parts(t, split(t), L, "777", reference=100_000)
    assert r["r_correct"] == 0
    assert r["reward"] == pytest.approx(FORMAT_BONUS)  # well formed, wrong, nothing else


def test_guess_without_reading_is_not_rewarded_over_reading():
    guess = "<answer>778</answer>"   # skips reading, wrong
    r_guess = reward_parts(guess, split(guess), L, "777", reference=100_000)
    r_good = reward_parts(GOOD, split(GOOD), L, "777", reference=100_000)
    assert r_good["reward"] > r_guess["reward"]


def test_broken_tags():
    t = GOOD.replace("</local>", "")
    r = reward_parts(t, split(t), L, "777", reference=100_000)
    assert r["r_format"] == -FORMAT_BONUS


def test_reading_less_scores_higher():
    lazy = '<focus magic_chunks="2">777</focus><answer>777</answer>'
    global_heavy = "<global>" + "let me look around the whole document " * 5 + "</global>" + GOOD
    a = reward_parts(lazy, split(lazy), L, "777", reference=100_000)
    b = reward_parts(global_heavy, split(global_heavy), L, "777", reference=100_000)
    assert a["r_correct"] == b["r_correct"] == 1
    assert a["r_efficiency"] > b["r_efficiency"]


def test_rambling_is_punished_with_a_fixed_reference():
    short = '<focus magic_chunks="2">777</focus><answer>777</answer>'
    long = '<focus magic_chunks="2">777</focus><local>' + "hmm " * 400 + "</local><answer>777</answer>"
    ref = 1000 * 60  # e.g. a 60 token vanilla answer on a 1000 token prompt
    a = reward_parts(short, split(short), L, "777", reference=ref)
    b = reward_parts(long, split(long), L, "777", reference=ref)
    assert a["r_efficiency"] > b["r_efficiency"]


def test_efficiency_clip():
    assert efficiency(50, 100) == 0.5
    assert efficiency(500, 100) == 0.0
    assert efficiency(0, 100) == 1.0
    assert efficiency(10, 0) == 0.0


def test_fallback_flag():
    r = reward_parts(GOOD, split(GOOD), L, "777")
    assert r["used_fallback_ref"]
