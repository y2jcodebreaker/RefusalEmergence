"""B1-v2 -- is the register table robust to the labeller, and what is Gemma's row?

    python3 b1_v2_registers.py            # CPU only, seconds

WHY THIS EXISTS (pre-registered in P1-Coupling-Not-Capability.md section 20, 2026-09-26, before
any label was computed). Two jobs, one script, because the first cannot be done without deciding
the second.

  1. Gemma-2-9B-it has no register row. B2 (2026-09-24) measured its sufficiency (97.0%) but
     predates the four-register table, so Figure 1 would have three full panels and one stub.
  2. B1's register table is `stance_of` v1 -- the labeller whose opening-only rule already
     retired A2's per-register result (O-184, O-199), and which A3-v2 showed moves 20 of 41
     Tulu-2 identity items to compliance in A3's pool (O-203). Section 3 of the paper LEADS on
     Tulu-2's identity 20 -> 21. If that fact is v1-specific, the lead result changes.

WHAT IT DOES NOT DO. It does not touch b1_registers.py or its frozen table, and asserts nothing
against them. It re-labels the same completions with both labellers and reports both.

RESIDUALS ARE NEVER ABSORBED. A WildGuard refusal that a labeller calls compliance/confusion is
reported in one of two buckets -- hand-audited-normative (the 86-item set b1_registers already
establishes) or UNAUDITED -- and never silently assigned to a register. B1 folded the audited
ones into 'normative'; that column is reproduced here as `with_audited` so the two tables are
comparable, beside the raw one.

PRE-REGISTERED PREDICTIONS (section 20, verbatim in v_verdict's docstring):
  V1  the inability invariant survives re-labelling  (>= 80% removed, both labellers)
  V2  Tulu-2's identity survivor fact survives v2     (ablated >= baseline, both >= 10)
  V3  no register but inability is cleanly removed where refusal does not collapse
  V4  Gemma is inability-dominant at baseline and no register gains under ablation

Output: results/b1_v2_registers_ANALYSIS.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Callable

from b1_registers import audited_residuals, wg_verdicts
from judge_bench import CONFUSION_SUBSTRINGS, label
from refusal_substring import truncate_at_turn
from runlog import RunRecord
from stance_v2 import stance_of_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("B1v2")

EXPERIMENT = "B1-v2"
QUESTION = ("Does B1's register table survive re-labelling with stance_of v2, and what is "
            "Gemma-2-9B-it's register row?")
R = "results"
REGISTERS = ("inability", "identity", "condemnation", "normative")
# Gemma is appended to B1's three. Its stem follows judge_wildguard's naming, like the others.
FAMILIES = {
    "zephyr_dpo": "zephyr_dpo_gen128",
    "tulu2_dpo": "tulu2_dpo_dpo_gen128",
    "olmo2_rlvr": "olmo2_e7_rlvr_gen128",
    "gemma2_it": "gemma2_it_it_gen128",
}
ARMS = ("baseline", "ablated")


def label_v2(text: str) -> str:
    """v2 register, with v1's confusion check kept so the two labellers differ in ONE respect."""
    if any(x in truncate_at_turn(text).lower() for x in CONFUSION_SUBSTRINGS):
        return "confusion"
    return stance_of_v2(text)


LABELLERS: dict[str, Callable[[str], str]] = {"v1": label, "v2": label_v2}


def count_registers(comps: list[str], wg: list[bool], labeller: Callable[[str], str],
                    audited: set[tuple[str, str, int]], fam: str, arm: str) -> dict:
    """Register counts among WildGuard refusals, with the residual split reported, not absorbed."""
    st: Counter = Counter()
    resid_audited, resid_unaudited = 0, []
    for i, (text, w) in enumerate(zip(comps, wg)):
        if not w:
            continue
        lab = labeller(text)
        if lab in ("compliance", "confusion"):
            if (fam, arm, i) in audited:
                resid_audited += 1
            else:
                resid_unaudited.append(i)
            continue
        st[lab] += 1
    raw = {r: st.get(r, 0) for r in REGISTERS}
    with_audited = dict(raw)
    with_audited["normative"] += resid_audited  # B1's rule, reproduced for comparability
    return {"raw": raw, "with_audited": with_audited, "n_refusals": int(sum(wg)),
            "residual_audited_normative": resid_audited,
            "residual_unaudited": len(resid_unaudited),
            "residual_unaudited_idx": resid_unaudited}


def removed_frac(baseline: int, ablated: int) -> float | None:
    """Fraction of a register's baseline count removed; None when there is nothing to remove."""
    return None if baseline == 0 else 1.0 - ablated / baseline


def v_verdict(fams: dict) -> dict:
    """The four pre-registered predictions of section 20. Pure, so a test can drive it.

    V1  In every family whose baseline inability count is >= 20, ablation removes >= 80% of it
        under BOTH labellers.
    V2  Tulu-2's identity count does not fall from baseline to ablated under v2
        (ablated >= baseline), with both >= 10.
    V3  No register other than inability is removed by >= 80% in a family where its own baseline
        count is >= 20, EXCEPT where total refusal also collapses (ablated n_refusals == 0).
    V4  Gemma's baseline is inability-dominant, and no register gains from baseline to ablated.
    """
    out: dict = {}

    v1_fail = []
    for fam, row in fams.items():
        for lv in ("v1", "v2"):
            b = row[lv]["baseline"]["with_audited"]["inability"]
            a = row[lv]["ablated"]["with_audited"]["inability"]
            if b >= 20 and (removed_frac(b, a) or 0.0) < 0.80:
                v1_fail.append(f"{fam}/{lv}: inability {b} -> {a}")
    out["V1"] = {"pass": not v1_fail, "detail": v1_fail or "inability >= 80% removed everywhere "
                 "its baseline reaches 20, under both labellers"}

    t = fams.get("tulu2_dpo", {}).get("v2")
    if t is None:
        out["V2"] = {"pass": False, "detail": "tulu2_dpo v2 row missing"}
    else:
        b = t["baseline"]["with_audited"]["identity"]
        a = t["ablated"]["with_audited"]["identity"]
        out["V2"] = {"pass": a >= b and b >= 10 and a >= 10,
                     "detail": f"Tulu-2 identity under v2: {b} -> {a}"}

    v3_fail = []
    for fam, row in fams.items():
        for lv in ("v1", "v2"):
            if row[lv]["ablated"]["n_refusals"] == 0:
                continue  # total collapse: register-specific reading unavailable (Zephyr)
            for reg in REGISTERS:
                if reg == "inability":
                    continue
                b = row[lv]["baseline"]["with_audited"][reg]
                a = row[lv]["ablated"]["with_audited"][reg]
                if b >= 20 and (removed_frac(b, a) or 0.0) >= 0.80:
                    v3_fail.append(f"{fam}/{lv}: {reg} {b} -> {a}")
    out["V3"] = {"pass": not v3_fail,
                 "detail": v3_fail or "inability is the only cleanly-removed register"}

    g = fams.get("gemma2_it")
    if g is None:
        out["V4"] = {"pass": False, "detail": "gemma2_it row missing"}
    else:
        det, ok = [], True
        for lv in ("v1", "v2"):
            b = g[lv]["baseline"]["with_audited"]
            a = g[lv]["ablated"]["with_audited"]
            dominant = max(b, key=lambda r: b[r])
            gains = [r for r in REGISTERS if a[r] > b[r]]
            ok = ok and dominant == "inability" and not gains
            det.append(f"{lv}: baseline dominant={dominant} {b}, ablated {a} "
                       f"(n={a and g[lv]['ablated']['n_refusals']}), gains={gains or 'none'}")
        out["V4"] = {"pass": ok, "detail": det,
                     "power_note": "Gemma's ablated arm has "
                                   f"{g['v1']['ablated']['n_refusals']} WildGuard refusals; "
                                   "'no register gains' is a weak test at this n."}
    return out


def main() -> None:
    audited = audited_residuals()
    fams: dict = {}
    for fam, stem in FAMILIES.items():
        arms = {arm: wg_verdicts(stem, arm) for arm in ARMS}
        fams[fam] = {lv: {arm: count_registers(c, wg, fn, audited, fam, arm)
                          for arm, (c, wg) in arms.items()}
                     for lv, fn in LABELLERS.items()}

    verdicts = v_verdict(fams)

    # Save before reporting (O-157): the JSON lands before any string that could raise.
    with open(f"{R}/b1_v2_registers_ANALYSIS.json", "w") as fh:
        json.dump({"what": QUESTION, "prereg": "P1-Coupling-Not-Capability.md section 20",
                   "families": fams, "verdicts": verdicts}, fh, indent=1)
    with RunRecord(EXPERIMENT, "b1_v2_registers.py", cfg=None, question=QUESTION,
                   notes="CPU only; re-labels B1's completions with v1 and v2 and adds Gemma") as r:
        for fam, row in fams.items():
            for lv in ("v1", "v2"):
                r.result(family=fam, labeller=lv,
                         baseline=row[lv]["baseline"]["with_audited"],
                         ablated=row[lv]["ablated"]["with_audited"],
                         residual_unaudited=(row[lv]["baseline"]["residual_unaudited"],
                                             row[lv]["ablated"]["residual_unaudited"]))
        for k, v in verdicts.items():
            r.result(**{k: v["pass"]})

    print("\n=== B1-v2: register table under both labellers ===")
    for fam, row in fams.items():
        print(f"\n{fam}")
        for lv in ("v1", "v2"):
            for arm in ARMS:
                d = row[lv][arm]
                cells = " ".join(f"{r[:4]}={d['with_audited'][r]:>3d}" for r in REGISTERS)
                print(f"  {lv} {arm:8s} n={d['n_refusals']:>3d}  {cells}"
                      f"  | resid audited={d['residual_audited_normative']} "
                      f"unaudited={d['residual_unaudited']}")
    print("\n=== pre-registered verdicts (section 20) ===")
    for k, v in verdicts.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['detail']}")
        if "power_note" in v:
            print(f"      {v['power_note']}")


if __name__ == "__main__":
    main()
