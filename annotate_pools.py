"""Item pools for the blind two-annotator labelling control. Pure, so it is testable.

Two controls in provenance.py are OPEN and both of them need human labels:

  B1, "human register labels"            -> TASK R: which register is this refusal in?
  A2, "blind full-completion human labels" -> TASK J: is this a refusal at all?

They are different questions on different item pools, and conflating them is the reason the
single "~300-400 items" estimate in the daily notes was never actionable. This module defines
each pool exactly, from the same frozen inputs the claims rest on.

WHY TASK J IS NOT JUST WILDGUARD'S MISSES. The claim-bearing cell for section 4 is the set
where our labeller says refusal and WildGuard says compliance (the residual gap). A sheet made
only of that cell would measure whether WildGuard undercounts, having assumed it does. The
reverse cell -- WildGuard says refusal, our labeller says compliance -- is sampled in the same
sheet at the same rate, so the instrument can report error in both directions or neither.

THE POOLS ARE DISJOINT FROM THE CALIBRATION SET BY ASSERTION, not by convention. The 19 items
in b1_v2_residual_audit.json carry author labels and are used to train the annotators; an item
that trains cannot also score, and build() raises if one appears in both.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Callable

import judge_bench as jb
import stance_v2
from refusal_substring import truncate_at_turn

logger = logging.getLogger(__name__)

RESULTS = "results"

# The four B1 families at 128 tokens, baseline and ablated. Same arms b1_v2_registers.py
# scores, so a human register table is comparable to the machine one cell for cell.
B1_FILES: dict[str, str] = {
    "zephyr_dpo": "zephyr_dpo_gen128_wildguard.json",
    "tulu2_dpo": "tulu2_dpo_dpo_gen128_wildguard.json",
    "olmo2_rlvr": "olmo2_e7_rlvr_gen128_wildguard.json",
    "gemma2_it": "gemma2_it_it_gen128_wildguard.json",
}

# Task J reads A2's pinned 5920, because that is the set section 4's claim is about.
# Task R must additionally see Gemma, whose arms A2_LATER_TAGS deliberately excludes -- the
# pin exists to stop A2's TABLE absorbing later runs, not to make the file unreadable. So the
# register pool is collected with the Gemma tag permitted and then filtered to B1_FILES,
# which keeps the pairing assertions in jb.collect() while adding the fourth family.
A2_PINNED_EXCLUDE: tuple[str, ...] = jb.A2_LATER_TAGS
B1_EXCLUDE: tuple[str, ...] = tuple(t for t in jb.A2_LATER_TAGS if t != "gemma2_it_")

REGISTERS = jb.REGISTERS
TASKS = ("R", "J")

# One spelling per family. jb.family_of reads the A2 filenames ('olmo2') and B1_FILES keys
# the arm ('olmo2_rlvr'); an item sampled for both controls would otherwise be counted under
# two different family names in the same table.
FAMILY_NORM: dict[str, str] = {"zephyr_dpo": "zephyr", "tulu2_dpo": "tulu2",
                               "olmo2_rlvr": "olmo2", "gemma2_it": "gemma2",
                               "olmo2_e7d": "olmo2", "olmo2_e7": "olmo2"}


def family_norm(name: str) -> str:
    return FAMILY_NORM.get(name, name)


def v2_label(text: str) -> str:
    """stance_of v2 with v1's confusion check, exactly as b1_v2_registers.label_v2 does."""
    if any(x in truncate_at_turn(text).lower() for x in jb.CONFUSION_SUBSTRINGS):
        return "confusion"
    return stance_v2.stance_of_v2(text)


def calibration_uids(path: str = f"{RESULTS}/b1_v2_residual_audit.json") -> set[tuple]:
    """(file, arm, i) of every item that already carries an author label.

    These train the annotators, so they must never score them. Returned as the provenance
    triple rather than a uid because the audit file predates this module's hashing.

    KEYED BY FILE, not by family, and the difference is not cosmetic. The audit records a
    B1 family key ('tulu2_dpo') and an arm ('baseline'), and those items exist only in the
    four gen128 arms; several results files normalise to the same family name, so a
    (family, arm, i) key matched 24 pool items where only 16 audited ones were present and
    over-excluded eight innocent completions from other arms. (family, arm, i) is not an
    item identity. (file, arm, i) is, and it is the key jb.collect() emits.

    Two earlier spellings of this exclusion failed in opposite directions -- one silently
    let all 19 back in, one quietly removed items it should not have -- which is why
    exclude_calibration() asserts its own effect instead of filtering and hoping."""
    if not os.path.exists(path):
        return set()
    d = json.load(open(path))
    out = set()
    for x in d.get("items", []):
        fname = B1_FILES.get(x["family"])
        if fname is None:
            raise AssertionError(f"audit family {x['family']!r} is not one of the four B1 "
                                 f"families; the calibration set cannot be located")
        out.add((fname, x["arm"], x["i"]))
    return out


def exclude_calibration(pool: list[dict], calib: set[tuple]) -> list[dict]:
    """Pool minus the author-labelled items, with the removal asserted.

    An item that trains the annotators cannot also score them. If the key spellings ever
    drift apart again this raises instead of returning a contaminated pool."""
    if not calib:
        return list(pool)
    def key(x: dict) -> tuple:
        return (x["file"], x["arm"], x["i"])

    present = {key(x) for x in pool} & calib
    out = [x for x in pool if key(x) not in calib]
    if len(pool) - len(out) != len(present):
        raise AssertionError(f"calibration exclusion removed {len(pool) - len(out)} items "
                             f"but {len(present)} unique keys were present -- the key is "
                             f"not an item identity")
    if not present:
        logger.warning("no calibration item is in this pool (%d items); check that the "
                       "family spellings still match", len(pool))
    return out


def _arm_family(file: str, arm: str) -> str:
    """Family for a B1 arm, or jb.family_of for an A2 arm. Normalised to one spelling."""
    for fam, fname in B1_FILES.items():
        if file == fname:
            return family_norm(fam)
    return family_norm(jb.family_of(file))


def load_items() -> tuple[list[dict], list[dict], list[str]]:
    """(task_J_items, task_R_items, skipped). Two collects, two different pins.

    The J items are asserted to be exactly A2's frozen set, so a drifted results/ directory
    stops the kit rather than quietly changing what section 4's control was computed on.
    This assertion is the reason p1e7h_ was found missing from A2_LATER_TAGS on 2026-10-19.
    """
    j_items, skipped = jb.collect(exclude=A2_PINNED_EXCLUDE)
    n_arms = len({(x["file"], x["arm"]) for x in j_items})
    if (len(j_items), n_arms) != (jb.A2_FROZEN_N_ITEMS, jb.A2_FROZEN_N_ARMS):
        raise SystemExit(
            f"A2's input set has drifted: {len(j_items)} items / {n_arms} arms against the "
            f"frozen {jb.A2_FROZEN_N_ITEMS} / {jb.A2_FROZEN_N_ARMS}. Task J would be sampled "
            f"from a different population than the claim it is meant to settle. Add the new "
            f"experiment's tag to judge_bench.A2_LATER_TAGS first.")
    r_items, skipped_r = jb.collect(exclude=B1_EXCLUDE)
    r_items = [x for x in r_items if x["file"] in set(B1_FILES.values())]
    missing = sorted(set(B1_FILES.values()) - {x["file"] for x in r_items})
    if missing:
        raise SystemExit(f"Task R is missing families: {missing}. The register table in "
                         f"section 3 has four of them; a human check of three is not the "
                         f"control provenance.py records as open.")
    return j_items, r_items, skipped + skipped_r


def task_r_pool(items: list[dict]) -> list[dict]:
    """Refusals in the four-family arms, with the stratum that must be balanced.

    An item enters if EITHER labeller or WildGuard calls it a refusal. Taking only the
    intersection would hand the annotators the easy cases and measure nothing; taking only
    v1's refusals would inherit v1's opening-only blind spot, which is the thing under test.
    """
    pool = []
    for x in items:
        if x["file"] not in B1_FILES.values():
            continue
        v1, v2 = x["register"], v2_label(x["text"])
        if not (v1 in REGISTERS or v2 in REGISTERS or x["wildguard"]):
            continue
        fam = _arm_family(x["file"], x["arm"])
        pool.append({**x, "task": "R", "family": fam, "v1": v1, "v2": v2,
                     "stratum": f"{fam}|{x['arm']}|{'agree' if v1 == v2 else 'split'}"})
    return pool


def task_j_pool(items: list[dict]) -> list[dict]:
    """The two cells where our labeller and WildGuard disagree about refusal itself.

    gap      v2 says refusal,    WildGuard says compliance  -- section 4's open half
    reverse  v2 says compliance, WildGuard says refusal     -- the other direction

    Both cells, so the sheet cannot only confirm the direction we already believe.
    """
    pool = []
    for x in items:
        v2 = v2_label(x["text"])
        if v2 in REGISTERS and not x["wildguard"]:
            cell = "gap"
        elif v2 == "compliance" and x["wildguard"]:
            cell = "reverse"
        else:
            continue
        fam = _arm_family(x["file"], x["arm"])
        pool.append({**x, "task": "J", "family": fam, "v1": x["register"], "v2": v2,
                     "cell": cell, "stratum": f"{cell}|{fam}"})
    return pool


def stratified(pool: list[dict], n: int, rng, key: Callable[[dict], str] = None) -> list[dict]:
    """n items spread over strata as evenly as the strata allow, deterministic in rng.

    Proportional allocation would hand almost the whole Task J sheet to OLMo 2, whose arms
    supply 84 of the gap and 185 of the reverse. The register claim is per family, so a
    sample that cannot speak about Tulu-2 identity is useless however large it is. Equal
    allocation with leftovers redistributed is the honest compromise, and the realised
    counts are recorded so the imbalance that remains is visible rather than implied.
    """
    key = key or (lambda x: x["stratum"])
    by: dict[str, list[dict]] = {}
    for x in pool:
        by.setdefault(key(x), []).append(x)
    for k in by:
        by[k] = sorted(by[k], key=lambda x: (x["file"], x["arm"], x["i"]))
        rng.shuffle(by[k])
    take: dict[str, int] = {k: 0 for k in by}
    while sum(take.values()) < n:
        room = [k for k in sorted(by) if take[k] < len(by[k])]
        if not room:
            break
        for k in room:
            if sum(take.values()) >= n:
                break
            take[k] += 1
    out = [x for k in sorted(by) for x in by[k][: take[k]]]
    rng.shuffle(out)
    return out


def describe(sample: list[dict], task: str = None) -> dict:
    """Realised strata, so the sheet's composition is a recorded number not a claim.

    `task` picks which pool's stratum to report for an item that serves both controls: its
    register stratum under R, its gap/reverse cell under J. Reporting one item under the
    other control's stratum label is how the first version of this table came out with
    register strata inside the task J column."""
    k = f"stratum_{task}" if task else "stratum"
    return {"n": len(sample),
            "by_stratum": dict(sorted(Counter(x.get(k, x["stratum"])
                                              for x in sample).items())),
            "by_family": dict(sorted(Counter(x["family"] for x in sample).items())),
            "v1_v2_split": sum(1 for x in sample if x["v1"] != x["v2"])}
