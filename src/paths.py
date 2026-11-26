"""Where outputs go.

Configs write paths like $LTF_ROOT/runs. LTF_ROOT is:
  - whatever you set it to, if you set it
  - Google Drive (/content/drive/MyDrive/learning-to-focus) if Drive is mounted on Colab
  - ./outputs otherwise (laptop, or Colab when the Drive mount fails, which it did for us
    with a second signed-in Google account)
"""

import os
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/learning-to-focus")


def root():
    if os.environ.get("LTF_ROOT"):
        return os.environ["LTF_ROOT"]
    if DRIVE.parent.exists():
        return str(DRIVE)
    return str(Path("outputs").resolve())


def expand(value):
    """Expand $LTF_ROOT (and other env vars) in a config value. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    os.environ.setdefault("LTF_ROOT", root())
    return os.path.expandvars(value)
