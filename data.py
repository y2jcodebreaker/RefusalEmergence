"""Harmful/harmless instruction splits — Arditi's, via a clone of andyrdt/refusal_direction.

Only `dataset/splits/*.json` is used; no code from that repo is imported.

Location is resolved by searching, not by one hardcoded path: the previous version
defaulted to a path on the author's laptop, which on any other machine produced an error
naming a directory that could not possibly exist there (hit on a pod, 2026-09-13).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

__all__ = ["load_instructions", "splits_dir", "assert_available", "SEARCH"]

# Checked in order. $ARDITI_REPO wins; the rest are where a clone plausibly sits.
SEARCH = (
    os.environ.get("ARDITI_REPO"),
    "./refusal_direction",
    "../refusal_direction",
    "/workspace/refusal_direction",
    "~/refusal_direction",
)


def splits_dir() -> Path | None:
    """First location that actually contains the splits, or None."""
    for cand in SEARCH:
        if not cand:
            continue
        p = Path(cand).expanduser() / "dataset" / "splits"
        if p.is_dir():
            return p
    return None


def assert_available() -> Path:
    """Fail fast, BEFORE any model is loaded. A missing clone costs nothing to detect and
    a full weight download to discover late."""
    p = splits_dir()
    if p is not None:
        return p
    tried = "\n".join(f"    {c}" for c in SEARCH if c)
    raise SystemExit(
        "Arditi harmful/harmless splits not found.\n"
        f"  ARDITI_REPO={os.environ.get('ARDITI_REPO') or '(unset)'}\n"
        f"  searched:\n{tried}\n\n"
        "  Fix:\n"
        "    git clone https://github.com/andyrdt/refusal_direction.git\n"
        "    export ARDITI_REPO=$PWD/refusal_direction\n"
        "  (only dataset/splits/*.json is read)")


def load_instructions(name: str) -> List[str]:
    """e.g. 'harmful_train', 'harmless_val'. Reads {name}.json (list of {instruction,...})."""
    base = assert_available()
    f = base / f"{name}.json"
    if not f.is_file():
        have = sorted(p.stem for p in base.glob("*.json"))
        raise SystemExit(f"split {name!r} not found in {base}. Available: {have}")
    with open(f) as fh:
        return [r["instruction"] for r in json.load(fh)]
