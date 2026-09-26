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
VARIANT_ORDER = ("v1", "v2", "v2+audit")


def hand_labels() -> dict[tuple[str, str, int], str]:
    """(family, arm, index) -> hand label, from the 2026-09-26 unblinded residual audit.

    Added AFTER V2 was scored and failed on machine v2 labels. It does not rescue that verdict
    -- V2 stands as FAILED -- it is reported beside it as new, dated evidence.
    """
    d = json.load(open(f"{R}/b1_v2_residual_audit.json"))
    return {(x["family"], x["arm"], x["i"]): x["label"] for x in d["items"]}


def count_registers(comps: list[str], wg: list[bool], labeller: Callable[[str], str],
                    audited: set[tuple[str, str, int]], fam: str, arm: str,
                    hand: dict[tuple[str, str, int], str] | None = None) -> dict:
    """Register counts among WildGuard refusals, with the residual split reported, not absorbed.

    `hand`, when given, resolves a residual to the register a human read it as -- or leaves it in
    the compliance bucket if the human read it as compliance.
    """
    st: Counter = Counter()
    resid_audited, resid_unaudited, hand_resolved = 0, [], 0
    for i, (text, w) in enumerate(zip(comps, wg)):
        if not w:
            continue
        lab = labeller(text)
        if lab in ("compliance", "confusion"):
            h = (hand or {}).get((fam, arm, i))
            if h in REGISTERS:
                st[h] += 1
                hand_resolved += 1
                continue
            if h == "compliance":
                continue  # human confirmed the labeller: genuinely not a refusal
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
            "residual_unaudited_idx": resid_unaudited,
            "hand_resolved": hand_resolved}


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

    for key, lv in (("V2", "v2"), ("V2b", "v2+audit")):
        t = fams.get("tulu2_dpo", {}).get(lv)
        if t is None:
            out[key] = {"pass": False, "detail": f"tulu2_dpo {lv} row missing"}
            continue
        b = t["baseline"]["with_audited"]["identity"]
        a = t["ablated"]["with_audited"]["identity"]
        out[key] = {"pass": a >= b and b >= 10 and a >= 10,
                    "detail": f"Tulu-2 identity under {lv}: {b} -> {a}"}
    # V2 is the pre-registered criterion and its verdict is final. V2b is the same rule applied
    # to hand-read residuals, scored AFTER V2 failed, and is reported as new evidence -- never as
    # a reason to reopen V2. Do not delete this note; it is what stops the rescue.
    out["V2b"]["status"] = ("POST-AUDIT, NOT A RESCUE. V2 stands at its recorded verdict. "
                            "The audit is results/b1_v2_residual_audit.json, unblinded.")

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
        # Informational, NOT part of V4's pre-registered scope (v1/v2 only): the audited variant.
        ab = g["v2+audit"]["baseline"]["with_audited"]
        aa = g["v2+audit"]["ablated"]["with_audited"]
        audit_gains = [f"{r} {ab[r]}->{aa[r]}" for r in REGISTERS if aa[r] > ab[r]]
        out["V4"] = {"pass": ok, "detail": det,
                     "power_note": "Gemma's ablated arm has "
                                   f"{g['v1']['ablated']['n_refusals']} WildGuard refusals; "
                                   "'no register gains' is a weak test at this n.",
                     "v2_audit_informational": (
                         f"Under v2+audit Gemma DOES gain: {audit_gains or 'nothing'}. Its four "
                         "residual refusals are all normative, so Gemma has a VESTIGIAL fallback, "
                         "not none. Reported because V4's v1/v2 pass would otherwise overstate "
                         "the 'no fallback' reading.")}
        out["DEFINITIONAL_WARNING"] = {
            "pass": None,
            "detail": ("Once inability goes to ~0, residual refusal IS the fallback register "
                       "count, so 'sufficiency tracks fallback capacity' is an IDENTITY up to the "
                       "baseline denominator, not an independent correlation. The non-definitional "
                       "content is: (a) inability is removed wherever it exists, (b) the fallback "
                       "register does not fall (Tulu-2 identity 17->19 under v2+audit), and (c) a "
                       "fallback can APPEAR rather than merely survive (OLMo 2 normative 2->64, "
                       "Gemma 1->4). Section 3 must be written on (a)-(c), never on the identity.")}
    return out


def main() -> None:
    audited = audited_residuals()
    hand = hand_labels()
    # Three columns: the two labellers as pre-registered, plus v2 with its residuals hand-read.
    variants: dict[str, tuple[Callable[[str], str], dict | None]] = {
        "v1": (label, None), "v2": (label_v2, None), "v2+audit": (label_v2, hand)}
    fams: dict = {}
    for fam, stem in FAMILIES.items():
        arms = {arm: wg_verdicts(stem, arm) for arm in ARMS}
        fams[fam] = {lv: {arm: count_registers(c, wg, fn, audited, fam, arm, h)
                          for arm, (c, wg) in arms.items()}
                     for lv, (fn, h) in variants.items()}

    verdicts = v_verdict(fams)

    # Save before reporting (O-157): the JSON lands before any string that could raise.
    with open(f"{R}/b1_v2_registers_ANALYSIS.json", "w") as fh:
        json.dump({"what": QUESTION, "prereg": "P1-Coupling-Not-Capability.md section 20",
                   "families": fams, "verdicts": verdicts}, fh, indent=1)
    with RunRecord(EXPERIMENT, "b1_v2_registers.py", cfg=None, question=QUESTION,
                   notes="CPU only; re-labels B1's completions with v1 and v2 and adds Gemma") as r:
        for fam, row in fams.items():
            for lv in VARIANT_ORDER:
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
        for lv in VARIANT_ORDER:
            for arm in ARMS:
                d = row[lv][arm]
                cells = " ".join(f"{r[:4]}={d['with_audited'][r]:>3d}" for r in REGISTERS)
                print(f"  {lv} {arm:8s} n={d['n_refusals']:>3d}  {cells}"
                      f"  | resid audited={d['residual_audited_normative']} "
                      f"unaudited={d['residual_unaudited']}")
    print("\n=== pre-registered verdicts (section 20) ===")
    for k, v in verdicts.items():
        tag = "NOTE" if v["pass"] is None else ("PASS" if v["pass"] else "FAIL")
        print(f"  {k}: {tag}  {v['detail']}")
        for extra in ("power_note", "v2_audit_informational", "status"):
            if extra in v:
                print(f"      {v[extra]}")


if __name__ == "__main__":
    main()
