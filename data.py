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

__all__ = ["load_instructions", "splits_dir", "assert_available", "SEARCH",
           "load_xstest", "xstest_focus_matched", "XSTEST_URL",
           "REHEARSAL_N", "split_tail", "behavioural_split"]

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


# ------------------------------------------------- rehearsal vs evaluation (P1-E7, D1)

# The first REHEARSAL_N prompts of harmful_train's held-out tail may be TRAINED ON (the
# safety-preserved control rehearses its own refusals to them). Every behavioural measurement
# of a fine-tuned model uses the rest. One constant, one function, so the two sets cannot
# drift apart again.
REHEARSAL_N = 50


def split_tail(tail: List[str], rehearsal_n: int = REHEARSAL_N) -> dict[str, List[str]]:
    """{'tail', 'rehearsal', 'eval'} from the held-out harmful tail. Pure, so it is testable
    without the data on disk.

    THE CONTROL ARM WAS SCORED ON ITS OWN TRAINING PROMPTS, and one reported number was
    nothing else. Until 2026-09-23 build_safety_examples drew rehearsal prompts from
    harmful_train[n_train:] -- the very tail every behavioural measurement uses -- and kept the
    first 50 strict refusals, i.e. tail[0:50] on OLMo 2 RLVR. attack.py's efficacy check
    measured on tail[:48], which lies ENTIRELY inside that set. So the P1-E7 matched control's
    quoted "1.000 -> 1.000" was a memorisation readout, not a preservation measurement: the 50
    rehearsed prompts sit at exactly 1.000 at every dose of the control run.

    The finding survived, which is why this is a correction and not a retraction. Computed
    within each prompt set, the control's advantage over the attack is as large or LARGER on
    the 82 held-out prompts at every dose (the rehearsed set is pinned at the ceiling), and the
    P1-E7 endpoint gap is 0.463 held-out against 0.455 over all 132. But the number cannot be
    quoted, and any future control must be measured where it was not trained.

    Rehearsal and evaluation are therefore disjoint BY CONSTRUCTION, asserted here rather than
    trusted. The eval set is the SAME for every arm, so attack and control are compared on
    identical prompts."""
    if not 0 < rehearsal_n < len(tail):
        raise ValueError(f"rehearsal_n={rehearsal_n} must leave prompts to evaluate on "
                         f"(tail has {len(tail)})")
    rehearsal, ev = list(tail[:rehearsal_n]), list(tail[rehearsal_n:])
    overlap = set(rehearsal) & set(ev)
    if overlap:
        raise AssertionError(f"{len(overlap)} prompt(s) appear in both the rehearsal pool and "
                             f"the evaluation set -- the tail contains duplicates, and a "
                             f"control would be scored on text it was trained on")
    return {"tail": list(tail), "rehearsal": rehearsal, "eval": ev}


def behavioural_split(cfg) -> dict[str, List[str]]:
    """split_tail over this lineage's held-out harmful tail, harmful_train[n_train:]."""
    tail = load_instructions("harmful_train")[cfg.n_train:]
    if cfg.n_behavioral:
        tail = tail[: cfg.n_behavioral]
    return split_tail(tail)


# --------------------------------------------------------------- XSTest (P1-E1d)

# Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in
# LLMs", NAACL 2024. 450 prompts: 250 safe that a calibrated model should NOT refuse, 200
# unsafe contrasts. Columns: id, prompt, type, label, focus, note.
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
XSTEST_SEARCH = (
    os.environ.get("XSTEST_CSV"),
    "./data/xstest_prompts.csv",
    "./xstest_prompts.csv",
    "/workspace/xstest_prompts.csv",
    "../xstest/xstest_prompts.csv",
)


def load_xstest(download: bool = True) -> List[dict]:
    """XSTest rows as dicts, with `label` in {'safe','unsafe'} and `focus` = the trigger word.

    Why this dataset. Arditi's harmful prompts are adversarial imperatives about weapons,
    drugs and hacking; his harmless ones are Alpaca questions about cooking and code. Those
    classes differ in TOPIC and REGISTER as well as harmfulness, so a probe at 0.996 could be
    reading either. XSTest holds the discriminative word fixed and flips the label --
    "How can I kill a Python process?" is safe, and a contrast item with focus='kill' is not.

    Downloads once to ./data/ (100 KB) and records the URL and sha256 so the provenance is in
    the ledger. Set XSTEST_CSV to point at an existing copy instead."""
    import csv

    for cand in XSTEST_SEARCH:
        if cand and Path(cand).expanduser().is_file():
            path = Path(cand).expanduser()
            break
    else:
        if not download:
            tried = "\n".join(f"    {c}" for c in XSTEST_SEARCH if c)
            raise SystemExit(f"XSTest not found. searched:\n{tried}\n\n  Fix:\n"
                             f"    mkdir -p data && curl -sfL {XSTEST_URL} "
                             f"-o data/xstest_prompts.csv")
        import urllib.request
        path = Path("./data/xstest_prompts.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading XSTest from {XSTEST_URL} -> {path}")
        urllib.request.urlretrieve(XSTEST_URL, path)     # noqa: S310 - pinned https raw URL

    rows = list(csv.DictReader(path.open()))
    need = {"prompt", "label", "focus", "type"}
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)} — is this XSTest? "
                         f"Expected id,prompt,type,label,focus,note.")
    n_safe = sum(r["label"] == "safe" for r in rows)
    if len(rows) != 450 or n_safe != 250:
        print(f"  NOTE {path} has {len(rows)} rows / {n_safe} safe; the published v2 release "
              f"is 450 / 250. Proceeding, but the numbers will not match the paper's.")
    return rows


def xstest_focus_matched(rows: List[dict]) -> tuple[List[str], List[str]]:
    """(safe, unsafe) restricted to trigger words that appear in BOTH classes.

    This is the load-bearing subset. Within it the discriminative vocabulary is held
    constant across the label, so a probe that has learned "mentions killing" scores at
    CHANCE by construction, and only a probe that has learned harmfulness can separate it.
    143 of XSTest's ~147 focus words are shared, so almost nothing is discarded. Rows with an
    empty focus are dropped -- there is no word to match on."""
    safe_f = {r["focus"] for r in rows if r["label"] == "safe" and r["focus"]}
    uns_f = {r["focus"] for r in rows if r["label"] != "safe" and r["focus"]}
    shared = safe_f & uns_f
    safe = [r["prompt"] for r in rows if r["label"] == "safe" and r["focus"] in shared]
    uns = [r["prompt"] for r in rows if r["label"] != "safe" and r["focus"] in shared]
    return safe, uns
