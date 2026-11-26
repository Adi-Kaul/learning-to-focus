# learning-to-focus

Can a small language model be *trained* to control its own attention?

[Ho et al. (2026)](https://arxiv.org/abs/2609.02737) showed that you can prompt a model to tag its own
reasoning with `<global>`, `<focus magic_chunks="K">` and `<local>`, and then have the inference engine
hide the parts of the document the model said it doesn't need. They call this Declarative Attention (DA).
Big models handle it well: Gemma-4-31B reads about 46% fewer KV tokens and loses 1.4 points of accuracy.
Small models don't. Gemma-4-E4B keeps only 29% of its normal accuracy under DA, and Qwen-3.5-4B keeps 64%.

The paper does everything with prompting and lists RL post-training as future work. This repo tries
the obvious next step on a ~4B model:

1. reproduce the small-model collapse on RULER needle-in-a-haystack (4K / 8K / 16K tokens)
2. SFT on the model's own good DA traces (rejection sampling)
3. GRPO with a reward for being right, well-formed, and reading less

and measures how much of the gap closes at each step.

Status: work in progress. Code and tests are in, experiments are next. Nothing below the line is a result yet.

---

## How it works (short version)

The document gets split into "magic chunks" of about 2K tokens and shown to the model as a fake tool
transcript (`get_magic_chunk` called once per chunk). Then the model is asked the question with the DA
instructions. While it writes, a small state machine watches for tags:

- `<global>` (default): everything is visible
- `<focus magic_chunks="3">`: only chunk 3 is visible
- `<local>` / `<answer>`: no chunks are visible, only the question and what the model already wrote

The first 16 prompt tokens (the attention sink) and everything from the question onwards are always visible.

The paper does the hiding inside vLLM. Here it's done in plain Hugging Face Transformers by changing the
attention mask every decode step, which only works cleanly for models where every layer is full
attention. That's why the main model is **Qwen3-4B** and not Qwen3.5-4B (which mixes in Gated DeltaNet layers).

## Layout

```
src/
  chunking.py   splits a document into magic chunks (paper's rules)
  prompts.py    vanilla and DA prompts, plus finding where each chunk sits in the token ids
  parse.py      the tag state machine
  masking.py    turns the state machine's mode into an attention mask
  decode.py     masked generation loop
  metrics.py    accuracy, tag adherence, attended tokens
  data.py       RULER NIAH loading + train/test split
  synth.py      tiny fake documents for the warm-up experiment
  generate.py   runs a model over a dataset, writes traces as JSONL
  train_sft.py, rewards.py, train_grpo.py
configs/        one yaml per experiment
tests/          pytest, runs on CPU
notebooks/      colab runner
```

## Running the tests

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Some tests download the Qwen3-0.6B tokenizer the first time.

## Credit

Prompt text and the tag rules come from the authors' reference implementation,
[itsnamgyu/declarative-attention](https://github.com/itsnamgyu/declarative-attention) (MIT). The RULER
data comes from their `dac-research/longbench_synthetic_v4_1` dataset on the Hugging Face Hub.
