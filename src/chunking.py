"""Split a long document into "magic chunks" the way the paper does.

The rules (section 3 of the paper, and segment.py in the authors' repo):
  - aim for ~2048 tokens per chunk, never go over 2560
  - prefer to cut at paragraph breaks, then newlines, then sentence ends,
    then commas/semicolons, then any whitespace
  - never cut inside a run of non-whitespace (a URL, a uuid, etc.)
  - chunks cover the whole document exactly, numbered 1..N

Everything is done with character offsets so that joining the chunk texts
gives back the exact original string. We only tokenize once.
"""

import re
from bisect import bisect_left
from dataclasses import dataclass

# coarsest to finest. each pattern keeps the delimiter attached to the piece before it
SPLIT_LEVELS = [
    re.compile(r"\n\s*\n"),     # paragraph break
    re.compile(r"\n"),          # newline
    re.compile(r"[.!?]+\s+"),   # end of sentence
    re.compile(r"[;:,]\s+"),    # clause
    re.compile(r"\s+"),         # any whitespace
]


@dataclass
class Chunk:
    id: int          # 1-based, like the paper
    text: str
    n_tokens: int
    start: int       # char offset into the original document
    end: int


def chunk_document(text, tokenizer, target=2048, cap=2560):
    if cap < target:
        raise ValueError("cap has to be >= target")
    if not text:
        return []
    if not getattr(tokenizer, "is_fast", False):
        # we need offset_mapping, which only the fast (rust) tokenizers give
        raise ValueError("need a fast tokenizer")

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tok_starts = [s for s, _ in enc["offset_mapping"]]

    def n_tokens(a, b):
        # a token "belongs" to whichever span its first character is in
        return bisect_left(tok_starts, b) - bisect_left(tok_starts, a)

    def cut(a, b, pattern):
        # split text[a:b] after every match of pattern
        pieces, prev = [], a
        for m in pattern.finditer(text, a, b):
            if m.end() > prev:
                pieces.append((prev, m.end()))
                prev = m.end()
        if prev < b:
            pieces.append((prev, b))
        return pieces

    def small_pieces(a, b, level=0):
        # break [a, b) down until every piece fits under the cap,
        # only going to a finer delimiter when the coarser one wasn't enough
        if n_tokens(a, b) <= cap or level == len(SPLIT_LEVELS):
            return [(a, b)]
        pieces = cut(a, b, SPLIT_LEVELS[level])
        if len(pieces) <= 1:
            return small_pieces(a, b, level + 1)
        out = []
        for pa, pb in pieces:
            out.extend(small_pieces(pa, pb, level + 1))
        return out

    # now glue the pieces back together greedily: start a new chunk if adding
    # the next piece would blow the cap, and close the chunk once it hits target
    spans = []
    chunk_start, chunk_tokens = None, 0
    for a, b in small_pieces(0, len(text)):
        n = n_tokens(a, b)
        if chunk_start is None:
            chunk_start = a
        elif chunk_tokens + n > cap:
            spans.append((chunk_start, a))
            chunk_start, chunk_tokens = a, 0
        chunk_tokens += n
        if chunk_tokens >= target:
            spans.append((chunk_start, b))
            chunk_start, chunk_tokens = None, 0
    if chunk_start is not None:
        spans.append((chunk_start, len(text)))

    return [Chunk(i + 1, text[a:b], n_tokens(a, b), a, b) for i, (a, b) in enumerate(spans)]


def find_chunk(chunks, char_pos):
    """Which chunk id holds this character position? (used to find the needle)"""
    for c in chunks:
        if c.start <= char_pos < c.end:
            return c.id
    return None


def chunks_containing(chunks, needle):
    """All chunk ids that contain `needle`. Usually one, but a needle can straddle
    a boundary in theory, and multikey distractors can repeat values."""
    ids = []
    doc = "".join(c.text for c in chunks)
    start = doc.find(needle)
    while start != -1:
        first = find_chunk(chunks, start)
        last = find_chunk(chunks, start + len(needle) - 1)
        for cid in range(first, last + 1):
            if cid not in ids:
                ids.append(cid)
        start = doc.find(needle, start + 1)
    return ids
