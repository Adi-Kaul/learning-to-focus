"""The tag state machine, plus an offline checker for finished traces.

Two different jobs live here:

1. TagMachine: runs *while* the model is generating. Feed it one token's text at a time and
   it tells you which mode the NEXT decode step is in. This decides the attention mask, so
   it copies the authors' engine behaviour closely (see their src/dattn/fsm/):
     - only look for tags when the new token contains a ">"
     - tags are matched at the very end of the text (up to 8 trailing chars allowed,
       because one token can be like '">\n' or similar)
     - the mode changes on the step AFTER the ">" token
     - <focus magic_chunks="..."> with a bad id list (empty, >3 ids, out of range, junk)
       doesn't switch modes, it just gets logged as a failed attempt
     - <answer> counts as <local>
     - in local mode we only look for the closing tag. a <focus> inside <local> is ignored
     - <global> / </global> tags do nothing at all, global is just the default

2. check_trace(): looks at a finished trace as a whole string and decides if the tags are
   well formed ("parse success" in our metrics) and pulls out the answer.
"""

import re

GLOBAL, FOCUS, LOCAL = "global", "focus", "local"

MAX_FOCUS_IDS = 3     # authors cap it at 3 chunks per focus
WINDOW = 500          # how much recent text we keep around to match tags against

# end-anchored: the tag has to be at the end of the text, give or take 8 chars
FOCUS_OPEN = re.compile(r"<focus[^>]*>.{0,8}$")
FOCUS_CLOSE = re.compile(r"</focus>.{0,8}$")
LOCAL_OPEN = re.compile(r"<(?:local|answer)>.{0,8}$")
LOCAL_CLOSE = re.compile(r"</(?:local|answer)>.{0,8}$")
FOCUS_IDS = re.compile(r'<focus\s+magic_chunks="([^"]*)">')


def parse_focus_ids(tag, n_chunks):
    """Returns (ids, error). error is "" if the tag is usable."""
    m = FOCUS_IDS.search(tag)
    if m is None:
        return (), "syntax_error"
    pieces = [p for p in re.split(r"[,\s]+", m.group(1).strip()) if p]
    if not pieces:
        return (), "empty_index_list"
    try:
        ids = tuple(int(p) for p in pieces)
    except ValueError:
        return (), "syntax_error"
    if len(ids) > MAX_FOCUS_IDS:
        return ids, "too_many_ids"
    if any(not 1 <= i <= n_chunks for i in ids):
        return ids, "out_of_range"
    return ids, ""


class TagMachine:
    def __init__(self, n_chunks):
        self.n_chunks = n_chunks
        self.mode = GLOBAL
        self.focus_ids = ()       # which chunks are visible right now (only meaningful in FOCUS)
        self.text = ""
        # bookkeeping for metrics
        self.focus_attempts = 0
        self.focus_successes = 0  # paper counts a success when </focus> closes, not when it opens
        self.focus_failures = {}  # reason -> count
        self.focused = []         # list of id tuples, one per successful focus open
        self.modes = []           # mode for every step fed so far (the mode that step was generated in)

    def feed(self, token_text):
        """Record that one token was generated (in the current mode), then update the mode
        for the next step. Returns True if the mode changed."""
        self.modes.append((self.mode, self.focus_ids))
        self.text = (self.text + token_text)[-WINDOW:]
        if ">" not in token_text:
            return False
        return self._check()

    def _check(self):
        tail = self.text[-WINDOW:].strip()
        if self.mode == GLOBAL:
            m = FOCUS_OPEN.search(tail)
            if m:
                self.focus_attempts += 1
                ids, err = parse_focus_ids(m.group(0), self.n_chunks)
                if err:
                    self.focus_failures[err] = self.focus_failures.get(err, 0) + 1
                    return False
                self.mode, self.focus_ids = FOCUS, ids
                self.focused.append(ids)
                return True
            if LOCAL_OPEN.search(tail):
                self.mode, self.focus_ids = LOCAL, ()
                return True
            return False
        if self.mode == FOCUS:
            if FOCUS_CLOSE.search(tail):
                self.focus_successes += 1
                self.mode, self.focus_ids = GLOBAL, ()
                return True
            return False
        # LOCAL
        if LOCAL_CLOSE.search(tail):
            self.mode, self.focus_ids = GLOBAL, ()
            return True
        return False


def replay(token_texts, n_chunks):
    """Run a finished trace through the machine token by token. Used by metrics to work out
    which mode every step was in, without needing the model."""
    m = TagMachine(n_chunks)
    for t in token_texts:
        m.feed(t)
    return m


# ---------------------------------------------------------------------------
# offline checks on a finished trace

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ANY_TAG = re.compile(r"</?(global|focus|local|answer)\b[^>]*>")


def extract_answer(text):
    """First <answer>...</answer> block, stripped. None if there isn't one.
    The paper grades the FIRST block, so a model that answers twice gets graded on the first."""
    m = ANSWER.search(text)
    return m.group(1).strip() if m else None


def check_trace(text, n_chunks):
    """Is the trace well formed? Returns (ok, list_of_problems).

    "Well formed" here means: every tag we know about opens and closes properly, they don't
    nest, focus tags have a usable chunk list, and there is exactly one <answer>.
    This is stricter than what the engine needs (the engine shrugs off a lot), which is
    the point. It's a measure of how cleanly the model follows the protocol.
    """
    problems = []
    open_tag = None
    n_answers = 0
    for m in ANY_TAG.finditer(text):
        raw, name = m.group(0), m.group(1)
        closing = raw.startswith("</")
        if not closing:
            if open_tag is not None:
                problems.append(f"<{name}> opened inside <{open_tag}>")
            open_tag = name
            if name == "focus":
                _, err = parse_focus_ids(raw, n_chunks)
                if err:
                    problems.append(f"bad focus tag {raw!r}: {err}")
            if name == "answer":
                n_answers += 1
        else:
            if open_tag != name:
                problems.append(f"</{name}> closes <{open_tag}>")
            open_tag = None
    if open_tag is not None:
        problems.append(f"<{open_tag}> never closed")
    if n_answers != 1:
        problems.append(f"{n_answers} <answer> blocks")
    return (not problems), problems
