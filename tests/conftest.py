import sys
from pathlib import Path

import pytest

# so `import src.whatever` works without installing anything
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TINY_MODEL)
