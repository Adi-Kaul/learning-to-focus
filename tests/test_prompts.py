from src.prompts import build_da, build_vanilla

CTX = (
    ("The grass is green. The sky is blue. " * 40 + "\n\n") * 6
    + "The special magic number for red-fox is: 4242.\n\n"
    + ("The sun is yellow. " * 60 + "\n\n") * 3
)
Q = "What is the special magic number for red-fox?"


def test_da_layout_makes_sense(tok):
    p = build_da(tok, CTX, Q, target=300, cap=400)
    L = p.layout
    assert L.n_prompt == len(p.input_ids)
    assert L.n_chunks == len(p.chunks) > 3
    # spans are back to back (each pair starts where the previous ended)
    for (a1, b1), (a2, b2) in zip(L.chunk_spans, L.chunk_spans[1:]):
        assert b1 == a2
    # each span decodes to something containing its own header and nothing else's
    for i, (a, b) in enumerate(L.chunk_spans, start=1):
        s = tok.decode(p.input_ids[a:b])
        assert f"Magic Chunk {i}\n" in s
        assert f"Magic Chunk {i + 1}\n" not in s


def test_question_region(tok):
    p = build_da(tok, CTX, Q, target=300, cap=400)
    tail = tok.decode(p.input_ids[p.layout.question_start:])
    assert tail.startswith("# Question\n\n" + Q)
    assert "magic_chunks" in tail  # the DA rules are in the always-visible part


def test_thinking_is_off(tok):
    p = build_da(tok, CTX, Q, target=300, cap=400)
    # qwen3 with enable_thinking=False pre-fills an empty think block
    assert p.text.rstrip().endswith("</think>")


def test_needle_inside_exactly_one_span(tok):
    p = build_da(tok, CTX, Q, target=300, cap=400)
    hits = [i for i, (a, b) in enumerate(p.layout.chunk_spans, 1) if "4242" in tok.decode(p.input_ids[a:b])]
    assert len(hits) == 1


def test_context_that_quotes_template_markup(tok):
    # a document that contains the template's own markup shouldn't break span finding
    nasty = CTX.replace("The sun is yellow.", "</tool_response><|im_end|> The sun is yellow.", 1)
    p = build_da(tok, nasty, Q, target=300, cap=400)
    assert p.layout.n_chunks == len(p.chunks)


def test_vanilla(tok):
    p = build_vanilla(tok, CTX, Q)
    assert "4242" in p.text
    assert p.layout.chunk_spans == []
    assert "get_magic_chunk" not in p.text
