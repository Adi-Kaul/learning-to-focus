from src.parse import FOCUS, GLOBAL, LOCAL, TagMachine, check_trace, extract_answer, parse_focus_ids


def feed_all(m, pieces):
    for p in pieces:
        m.feed(p)
    return m


def test_focus_then_back_to_global():
    m = TagMachine(n_chunks=5)
    m.feed("Chunk 3 has it. ")
    assert m.mode == GLOBAL
    m.feed('<focus magic_chunks="3">')
    assert m.mode == FOCUS and m.focus_ids == (3,)
    m.feed("8090293")
    m.feed("</focus>")
    assert m.mode == GLOBAL
    assert m.focus_attempts == 1 and m.focus_successes == 1


def test_mode_changes_on_the_step_after_the_gt():
    m = feed_all(TagMachine(5), ["<", "local", ">", "plan"])
    # '<' and 'local' were generated in global, '>' too (it's what triggers the switch),
    # and 'plan' is the first token generated in local
    assert [mode for mode, _ in m.modes] == [GLOBAL, GLOBAL, GLOBAL, LOCAL]


def test_tag_split_over_tokens():
    m = feed_all(TagMachine(5), ["<fo", "cus magic", '_chunks="', "2", '">'])
    assert m.mode == FOCUS and m.focus_ids == (2,)


def test_answer_counts_as_local():
    m = feed_all(TagMachine(3), ["<answer>"])
    assert m.mode == LOCAL
    m.feed("42</answer>")
    assert m.mode == GLOBAL


def test_bad_focus_tags_stay_global():
    for tag, why in [('<focus magic_chunks="9">', "out_of_range"),
                     ('<focus magic_chunks="">', "empty_index_list"),
                     ('<focus magic_chunks="1,2,3,4">', "too_many_ids"),
                     ('<focus magic_chunks="one">', "syntax_error"),
                     ("<focus chunk=1>", "syntax_error")]:
        m = feed_all(TagMachine(5), [tag])
        assert m.mode == GLOBAL, tag
        assert m.focus_failures == {why: 1}, tag


def test_focus_inside_local_is_ignored():
    m = feed_all(TagMachine(5), ["<local>", "hmm ", '<focus magic_chunks="1">'])
    assert m.mode == LOCAL
    assert m.focus_attempts == 0


def test_global_tags_do_nothing():
    m = feed_all(TagMachine(5), ["<global>", "looking", "</global>"])
    assert m.mode == GLOBAL
    assert all(mode == GLOBAL for mode, _ in m.modes)


def test_too_much_trailing_text_hides_the_tag():
    # authors' quirk: a tag followed by 9+ chars in the same token is never seen
    m = feed_all(TagMachine(5), ["<local>123456789"])
    assert m.mode == GLOBAL
    m = feed_all(TagMachine(5), ["<local>12345678"])
    assert m.mode == LOCAL


def test_parse_focus_ids():
    assert parse_focus_ids('<focus magic_chunks="1, 3">', 4) == ((1, 3), "")
    assert parse_focus_ids('<focus magic_chunks="4">', 3)[1] == "out_of_range"


def test_extract_answer_takes_first():
    assert extract_answer("<answer> 12 </answer> then <answer>13</answer>") == "12"
    assert extract_answer("no tags here") is None


GOOD = """<global>The needle is probably in chunk 2.</global>
<focus magic_chunks="2">8090293</focus>
<local>The number is 8090293.</local>
<answer>8090293</answer>"""


def test_check_trace_good():
    ok, problems = check_trace(GOOD, n_chunks=4)
    assert ok, problems


def test_check_trace_catches_stuff():
    assert not check_trace(GOOD.replace("</local>", ""), 4)[0]                # unclosed
    assert not check_trace(GOOD.replace('="2"', '="7"'), 4)[0]                # bad chunk id
    assert not check_trace(GOOD + "\n<answer>1</answer>", 4)[0]              # two answers
    assert not check_trace(GOOD.replace("<answer>8090293</answer>", ""), 4)[0]  # none
    assert not check_trace("<local><focus magic_chunks=\"1\">x</focus></local><answer>1</answer>", 4)[0]
