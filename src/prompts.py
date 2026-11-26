"""Building the vanilla and DA prompts, and working out where everything landed in the token ids.

The DA prompt is a fake tool conversation, same as the paper's headline format:

    system:     "You are a helpful assistant."   (plus the tool schema, the template puts it here)
    user:       "A question will follow about a long document. First, retrieve it..."
    assistant:  get_magic_chunk(id="1")
    tool:       "Magic Chunk 1\n<text>"
    ... one call/response pair per chunk ...
    user:       "# Question\n\n<question>\n\n# Instructions ..." (the DA rules)

For masking we need to know, in token positions:
  - sink: the first 16 tokens, always visible
  - question_start: where the final user turn's "# Question" header starts. From there
    to the end of the prompt is always visible (plus everything the model generates)
  - chunk_spans[k]: the tokens of the k-th call/response pair. These are what <focus> reveals.

Everything between the sink and question_start that isn't a focused chunk is hidden in
focus/local mode. That includes the bootstrap message and the tool schema, which is what
the authors do too (their base_mask is just sink + open tail).
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking import chunk_document

TEMPLATES = Path(__file__).parent / "templates"
DA_INSTRUCTIONS = (TEMPLATES / "da_instructions.md").read_text()
VANILLA_TEMPLATE = (TEMPLATES / "vanilla.md").read_text()

SYSTEM = "You are a helpful assistant."
SINK_TOKENS = 16

BOOTSTRAP = (
    "A question will follow about a long document. First, retrieve the document "
    "using the `get_magic_chunk` tool, one magic chunk at a time, in order from "
    'magic chunk id "1", until every magic chunk has been retrieved.'
)

GET_MAGIC_CHUNK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_magic_chunk",
        "description": (
            "Retrieve one magic chunk of the user's document by its 1-indexed "
            "id. The document is split into consecutive magic chunks, which are "
            "arbitrary retrieval units and do not align with the document's own "
            "sections or chapters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": 'The magic chunk id to retrieve (e.g. "1", "2", ...).'},
            },
            "required": ["id"],
        },
    },
}

# both templates have "# Question" right before {question}. we find the LAST one in the
# rendered prompt, since the document itself could contain that string
QUESTION_MARKER = "# Question\n"


@dataclass
class Layout:
    """Where things are in the prompt, in token positions. This is all masking needs."""
    n_prompt: int
    question_start: int
    chunk_spans: list = field(default_factory=list)  # [(start, end)] per chunk, end exclusive
    sink: int = SINK_TOKENS

    @property
    def n_chunks(self):
        return len(self.chunk_spans)

    def to_dict(self):
        return {"n_prompt": self.n_prompt, "question_start": self.question_start,
                "chunk_spans": [list(s) for s in self.chunk_spans], "sink": self.sink}

    @classmethod
    def from_dict(cls, d):
        return cls(d["n_prompt"], d["question_start"], [tuple(s) for s in d["chunk_spans"]], d.get("sink", SINK_TOKENS))


@dataclass
class Prompt:
    text: str
    input_ids: list
    layout: Layout
    chunks: list            # Chunk objects (empty for vanilla)
    messages: list


def _call_and_reply(chunk_id, text):
    call_id = f"call_{chunk_id}"
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            # dict, not a json string. Qwen templates do `arguments | tojson` themselves
            "function": {"name": "get_magic_chunk", "arguments": {"id": str(chunk_id)}},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": f"Magic Chunk {chunk_id}\n{text}"},
    ]


def da_messages(chunks, question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": BOOTSTRAP}]
    for c in chunks:
        msgs += _call_and_reply(c.id, c.text)
    msgs.append({"role": "user", "content": DA_INSTRUCTIONS.format(question=question)})
    return msgs


def vanilla_messages(context, question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": VANILLA_TEMPLATE.format(context=context, question=question)},
    ]


def _render(tokenizer, messages, tools):
    # enable_thinking=False is important for Qwen3: the paper found models ignore the
    # protocol inside <think> blocks, so thinking is off everywhere
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_vanilla(tokenizer, context, question):
    msgs = vanilla_messages(context, question)
    text = _render(tokenizer, msgs, None)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [s for s, _ in enc["offset_mapping"]]
    q = bisect_left(starts, text.rfind(QUESTION_MARKER))
    return Prompt(text, enc["input_ids"], Layout(len(starts), q, []), [], msgs)


def build_da(tokenizer, context, question, chunks=None, target=2048, cap=2560):
    """Chunk the context (unless chunks are given) and build the DA prompt + its layout."""
    if chunks is None:
        chunks = chunk_document(context, tokenizer, target=target, cap=cap)
    msgs = da_messages(chunks, question)
    tools = [GET_MAGIC_CHUNK_TOOL]
    text = _render(tokenizer, msgs, tools)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [s for s, _ in enc["offset_mapping"]]

    # Finding the chunk boundaries. Instead of regexing for template markup (which breaks
    # if the document happens to contain "<tool_response>"), render the conversation with
    # only the first k pairs, always keeping the final question turn. Two renders with
    # k and k+1 pairs agree up to the point where pair k+1 begins, so the common prefix
    # tells us the boundary. This is the same trick the authors use.
    head = msgs[:2]
    pairs = msgs[2:-1]
    question_turn = msgs[-1]
    n = len(chunks)
    renders = [_render(tokenizer, head + pairs[: 2 * k] + [question_turn], tools) for k in range(n)]
    renders.append(text)
    # one extra fake pair at the end so the last real pair gets a boundary the same way
    renders.append(_render(tokenizer, head + pairs + _call_and_reply(n + 1, "x") + [question_turn], tools))

    char_bounds = [_common_prefix_len(renders[k], renders[k + 1]) for k in range(n + 1)]
    tok_bounds = [bisect_left(starts, c) for c in char_bounds]
    spans = [(tok_bounds[k], tok_bounds[k + 1]) for k in range(n)]

    q = bisect_left(starts, text.rfind(QUESTION_MARKER))
    layout = Layout(len(starts), q, spans)
    _check_layout(layout, text, enc, chunks)
    return Prompt(text, enc["input_ids"], layout, chunks, msgs)


def _check_layout(layout, text, enc, chunks):
    # cheap paranoia. if any of these fire the masking would silently hide the wrong stuff
    assert layout.sink <= layout.question_start < layout.n_prompt, "question marker in a weird place"
    prev_end = layout.sink
    for (a, b), c in zip(layout.chunk_spans, chunks):
        assert prev_end <= a < b <= layout.question_start, f"chunk {c.id} span {a, b} out of order"
        prev_end = b
        span_text = text[enc["offset_mapping"][a][0]:enc["offset_mapping"][b - 1][1]]
        assert f"Magic Chunk {c.id}" in span_text, f"chunk {c.id} header not inside its span"
        # the chunk text itself should be in there too (checking a bit of it is enough)
        assert c.text[:50] in span_text and c.text[-50:] in span_text, f"chunk {c.id} body not in span"
