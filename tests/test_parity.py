"""Check our state machine + attended token count against the authors' own recorded cases.

tests/fixtures/authors_fsm_cases.jsonl is copied from their repo
(itsnamgyu/declarative-attention, tests/fixtures/fsm/cases.jsonl, MIT). It has 329 traces
(real model outputs + lots of deliberately broken ones) with the mode at every step and
the final metrics, produced by running their reference implementation.

If this passes, our masking decisions and our efficiency numbers are the paper's.
"""
import json
from pathlib import Path

import pytest

from src.metrics import attended_tokens
from src.parse import TagMachine
from src.prompts import Layout

FIXTURES = Path(__file__).parent / "fixtures" / "authors_fsm_cases.jsonl"
rows = [json.loads(line) for line in FIXTURES.open()]
header, cases = rows[0], rows[1:]

FAIL_NAMES = {  # their reason names -> ours
    "syntax_error": "syntax_error",
    "empty_index_list": "empty_index_list",
    "more_than_three_indices": "too_many_ids",
    "out_of_range_index": "out_of_range",
}


def layout_for(case):
    d = header["layouts_by_id"][case["layout_id"]]
    return Layout(d["n_prompt_tokens"], d["question_start"], [tuple(s) for s in d["segments"]], d["sink"]), d


@pytest.mark.parametrize("case", cases, ids=[c["case_id"] for c in cases])
def test_matches_authors(case):
    layout, d = layout_for(case)
    if layout.n_chunks == 0:
        pytest.skip("no-chunk layout (format error path), we never build those")
    m = TagMachine(layout.n_chunks)
    m.text = d["prompt_tail"][-500:]  # they seed the scan window with the end of the prompt

    # mode after each token, to compare against their event list
    after = []
    for t in case["token_texts"]:
        m.feed(t)
        after.append(m.mode)

    expected, cur = [], "global"
    events = {e["step"]: e["mode"].lower() for e in case["events"]}
    for step in range(len(case["token_texts"])):
        cur = events.get(step, cur)
        expected.append(cur)
    assert after == expected

    met = case["metrics"]
    assert m.focus_attempts == met["focus_attempts"]
    assert m.focus_successes == met["focus_successes"]
    for theirs, ours in FAIL_NAMES.items():
        assert m.focus_failures.get(ours, 0) == met[f"focus_fail_{theirs}"], theirs
    assert attended_tokens(layout, m.modes) == met["attended_tokens"]
