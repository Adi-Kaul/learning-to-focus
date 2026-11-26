import random

from src.chunking import chunk_document, chunks_containing, find_chunk


def make_doc(n_paragraphs=60, seed=0):
    rng = random.Random(seed)
    words = "the grass is green and the sky is blue while the sun is yellow".split()
    paras = []
    for _ in range(n_paragraphs):
        sents = []
        for _ in range(rng.randint(3, 8)):
            sents.append(" ".join(rng.choice(words) for _ in range(rng.randint(6, 20))).capitalize() + ".")
        paras.append(" ".join(sents))
    return "\n\n".join(paras)


def test_chunks_cover_document_exactly(tok):
    doc = make_doc()
    chunks = chunk_document(doc, tok, target=200, cap=260)
    assert "".join(c.text for c in chunks) == doc
    assert [c.id for c in chunks] == list(range(1, len(chunks) + 1))
    # spans line up end to end
    assert chunks[0].start == 0 and chunks[-1].end == len(doc)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start


def test_respects_cap(tok):
    chunks = chunk_document(make_doc(), tok, target=200, cap=260)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 260 for c in chunks)


def test_short_doc_is_one_chunk(tok):
    chunks = chunk_document("Just one short sentence.", tok)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."


def test_prefers_paragraph_breaks(tok):
    # every paragraph is ~ 50 tokens, so with target 100 every cut should land on a "\n\n"
    doc = make_doc(n_paragraphs=30, seed=3)
    chunks = chunk_document(doc, tok, target=100, cap=400)
    for c in chunks[:-1]:
        assert c.text.endswith("\n\n"), repr(c.text[-20:])


def test_never_splits_a_long_word(tok):
    blob = "x" * 3000  # no whitespace at all, way over the cap
    doc = "intro words here.\n\n" + blob + "\n\noutro words here."
    chunks = chunk_document(doc, tok, target=50, cap=60)
    assert any(blob in c.text for c in chunks)
    assert "".join(c.text for c in chunks) == doc


def test_empty(tok):
    assert chunk_document("", tok) == []


def test_find_needle_chunk(tok):
    doc = make_doc(seed=5)
    needle = "The special magic number for blue-whale is: 1234567."
    pos = len(doc) // 2
    pos = doc.index("\n\n", pos) + 2
    doc = doc[:pos] + needle + " " + doc[pos:]
    chunks = chunk_document(doc, tok, target=200, cap=260)
    ids = chunks_containing(chunks, "1234567")
    assert len(ids) == 1
    assert "1234567" in chunks[ids[0] - 1].text
    assert find_chunk(chunks, doc.index("1234567")) == ids[0]
