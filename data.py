"""Harmful/harmless instruction splits — reuse Arditi's (via ARDITI_REPO), same as e01."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

__all__ = ["load_instructions"]

_ARDITI = Path(os.environ.get("ARDITI_REPO") or
               "/Users/lichking/Documents/probe_repos/probe4_arditi_refusal_direction").expanduser()
_SPLITS = _ARDITI / "dataset" / "splits"


def load_instructions(name: str) -> List[str]:
    """e.g. 'harmful_train', 'harmless_val'. Reads {name}.json (list of {instruction,...})."""
    if not _SPLITS.is_dir():
        raise FileNotFoundError(
            f"Arditi splits not found: {_SPLITS}\n"
            f"  export ARDITI_REPO=/path/to/refusal_direction   (the cloned repo)")
    with open(_SPLITS / f"{name}.json") as f:
        return [r["instruction"] for r in json.load(f)]
