from src.failures import categorize

LAYOUT = {"n_prompt": 1000, "question_start": 900, "chunk_spans": [[100, 400], [400, 700], [700, 890]], "sink": 16}


def row(text, finished=True, gold="777", needle=(2,)):
    return {"text": text, "gold": gold, "finished": finished, "layout": LAYOUT, "needle_chunks": list(needle)}


def test_correct_is_none():
    assert categorize(row('<focus magic_chunks="2">777</focus><answer>777</answer>')) is None


def test_categories():
    assert categorize(row("<global>looking", finished=False)) == "ran_out"
    assert categorize(row("I think it's 777")) == "no_answer"
    assert categorize(row("<local>hmm<answer>1</answer>")) == "broken_tags"
    assert categorize(row("<answer>March 14, 2024</answer>")) == "never_focused"
    assert categorize(row('<focus magic_chunks="1">5</focus><answer>5</answer>')) == "wrong_chunk"
    assert categorize(row('<focus magic_chunks="2">778</focus><answer>778</answer>')) == "bad_extraction"
    assert categorize(row('<focus magic_chunks="2">777</focus><local>ok</local><answer>77</answer>')) == "lost_after_focus"
