"""The important correctness test: masked step-by-step decoding == one forward pass with the
training mask. If these disagree, either training or evaluation is masking the wrong thing.

Uses Qwen3-0.6B in float32 on CPU, takes ~30s.
"""
import pytest
import torch

from src.decode import generate
from src.masking import training_mask
from src.parse import replay
from src.prompts import build_da

CTX = (("The grass is green. The sky is blue. " * 30 + "\n\n") * 2
       + "The special magic number for red-fox is: 4242.\n\n"
       + ("The sun is yellow. " * 40 + "\n\n") * 2)
TRACE = ('<global>Chunk 2.</global>\n<focus magic_chunks="2">4242</focus>\n'
         '<local>So 4242.</local>\n<answer>4242</answer>')


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()


def teacher_forced_logprobs(model, prompt, cids, masked):
    full = torch.tensor([prompt.input_ids + cids])
    kwargs = {}
    if masked:
        m = replay(prompt._texts, prompt.layout.n_chunks)
        kwargs["attention_mask"] = training_mask(prompt.layout, m.modes, (m.mode, m.focus_ids))[None, None]
    with torch.no_grad():
        logits = model(input_ids=full, **kwargs).logits
    P = prompt.layout.n_prompt
    lp = torch.log_softmax(logits[0, P - 1:-1].float(), -1)
    return lp.gather(1, torch.tensor(cids)[:, None]).squeeze(1)


@pytest.fixture(scope="module")
def setup(tok):
    p = build_da(tok, CTX, "What is the special magic number for red-fox?", target=250, cap=320)
    cids = tok(TRACE, add_special_tokens=False)["input_ids"]
    p._texts = [tok.decode([i]) for i in cids]
    return p, cids


def test_masked_loop_matches_training_mask(model, tok, setup):
    p, cids = setup
    ref = teacher_forced_logprobs(model, p, cids, masked=True)
    tr = generate(model, tok, [p], masked=True, max_new_tokens=len(cids) + 1, force=[cids])[0]
    assert tr.token_ids == cids
    assert torch.allclose(torch.tensor(tr.logprobs), ref, atol=1e-3)


def test_unmasked_loop_matches_plain_forward(model, tok, setup):
    p, cids = setup
    ref = teacher_forced_logprobs(model, p, cids, masked=False)
    tr = generate(model, tok, [p], masked=False, max_new_tokens=len(cids) + 1, force=[cids])[0]
    assert torch.allclose(torch.tensor(tr.logprobs), ref, atol=1e-3)


def test_mask_actually_changes_something(model, tok, setup):
    p, cids = setup
    a = teacher_forced_logprobs(model, p, cids, masked=True)
    b = teacher_forced_logprobs(model, p, cids, masked=False)
    assert (a - b).abs().max() > 0.1


def test_batching_with_padding_matches_single(model, tok, setup):
    p, cids = setup
    # second prompt is shorter so it gets left padded. padding must not change anything
    p2 = build_da(tok, CTX[:1500], "What is the special magic number for red-fox?", target=250, cap=320)
    c2 = tok('<local>Not sure.</local><answer>1</answer>', add_special_tokens=False)["input_ids"]
    both = generate(model, tok, [p, p2], masked=True, max_new_tokens=len(cids) + 1, force=[cids, c2])
    alone = generate(model, tok, [p2], masked=True, max_new_tokens=len(c2) + 1, force=[c2])[0]
    single = generate(model, tok, [p], masked=True, max_new_tokens=len(cids) + 1, force=[cids])[0]
    assert both[1].token_ids == c2
    assert torch.allclose(torch.tensor(both[1].logprobs), torch.tensor(alone.logprobs), atol=1e-3)
    assert torch.allclose(torch.tensor(both[0].logprobs), torch.tensor(single.logprobs), atol=1e-3)
