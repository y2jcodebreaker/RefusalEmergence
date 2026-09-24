"""B1 -- regenerate the three-family sufficiency and register table from stored files, and assert it.

    python b1_registers.py                # CPU only, seconds

WHY THIS EXISTS. B1 is a headline claim (H2 in the claim map): directional ablation removes the
INABILITY register, and what survives is the model's FALLBACK register, so sufficiency tracks
fallback capacity (Zephyr-DPO 100% > Tulu-2 70% > OLMo 2 51% at 128 tokens). Its table,
results/b1_three_family_128_ANALYSIS.json, was written by hand (commit 6f945fc) -- no script
produced it. Found 2026-09-23 while building the claim map. A headline number nobody can
regenerate is a provenance hole, so this script regenerates every number in it from the
stored completions and WildGuard verdicts, and fails loudly if any differs.

HOW EACH NUMBER IS DERIVED.
  WildGuard verdicts   reconstructed exactly as judge_bench.py does (stored disagreement indices,
                       asserted against the stored rate), paired by judge_bench.source_for.
  sufficiency          removed = 1 - ablated/baseline WildGuard refusal COUNT, per family.
  register counts      judge_bench.label() (stance_of v1) on every WildGuard refusal.

THE ONE RULE THAT IS NOT MECHANICAL, AND WHY IT HOLDS. stance_of v1 misses prohibitive and
redirect phrasings of the normative register (O-184), so 24 WildGuard refusals across the six
arms come out 'compliance'. B1 counted them as normative. That is correct ONLY if each one is a
genuine normative refusal, and each one has been read:
  - 18 are OLMo 2 ablated items that were already WildGuard refusals at 48 tokens, all 80 of
    which were hand-read against their prompts (olmo2_e7_ablated_HANDAUDIT.json). The 128-token
    completion extends the 48-token one exactly (greedy prefix property, 132/132).
  - 6 were never read by anyone until 2026-09-23 (results/b1_residual_audit.json): all six are
    genuine refusals in the broad normative register (redirect, subversion).
An unaudited residual item makes the script refuse to assign it: it reports it, and the
assertion against the frozen table fails. The rule cannot silently cover new data.

CONFIDENCE INTERVALS are NOT asserted. The frozen table's intervals ([83,100], [55,80],
[40,60]) do not record their method. This script reports two of its own, each labelled: a
bootstrap over PROMPTS (paired, 10,000 draws, seed 0), which is degenerate when the ablated arm
has zero refusals (Zephyr: every resample says 100%), and a Wilson interval on the survival
count (ablated refusals of baseline refusals), which is not. Neither is substituted for the
frozen intervals without saying so.

Output: results/b1_registers_ANALYSIS.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np

from judge_bench import label, load_completions, source_for
from refusal_substring import is_refusal_strict, truncate_at_turn
from runlog import RunRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("B1")

EXPERIMENT = "B1"
QUESTION = ("Does B1's hand-written three-family table regenerate exactly from the stored "
            "completions and WildGuard verdicts?")
R = "results"
FAMILIES = {"zephyr_dpo": "zephyr_dpo_gen128", "tulu2_dpo": "tulu2_dpo_dpo_gen128",
            "olmo2_rlvr": "olmo2_e7_rlvr_gen128"}
REGISTERS = ("inability", "identity", "condemnation", "normative")
# What the frozen table asserts (b1_three_family_128_ANALYSIS.json, RESULTS.md B1).
FROZEN = {
    "zephyr_dpo": {"baseline": 0.227, "ablated": 0.000, "n": (30, 0),
                   "stances": {"baseline": {"inability": 2, "identity": 0, "condemnation": 24,
                                            "normative": 4},
                               "ablated": {"inability": 0, "identity": 0, "condemnation": 0,
                                           "normative": 0}}},
    "tulu2_dpo": {"baseline": 0.758, "ablated": 0.227, "n": (100, 30),
                  "stances": {"baseline": {"inability": 78, "identity": 20, "condemnation": 0,
                                           "normative": 2},
                              "ablated": {"inability": 8, "identity": 21, "condemnation": 0,
                                          "normative": 1}}},
    "olmo2_rlvr": {"baseline": 0.985, "ablated": 0.485, "n": (130, 64),
                   "stances": {"baseline": {"inability": 128, "identity": 0, "condemnation": 0,
                                            "normative": 2},
                               "ablated": {"inability": 0, "identity": 0, "condemnation": 0,
                                           "normative": 64}}},
}


def wg_verdicts(stem: str, arm: str) -> tuple[list[str], list[bool]]:
    """Completions and WildGuard verdicts for one arm, reconstructed and asserted."""
    path = f"{R}/{stem}_wildguard.json"
    d = json.load(open(path))[arm]
    comps: dict = {}
    for cand in source_for(path):
        comps = load_completions(cand)
        if comps:
            break
    if not comps:
        raise SystemExit(f"{stem}: no completions on disk")
    c = comps[arm]
    sub = [is_refusal_strict(truncate_at_turn(x)) for x in c]
    dis = set(d.get("disagreements") or [])
    wg = [(not sub[i]) if i in dis else sub[i] for i in range(len(c))]
    if abs(sum(wg) / len(wg) - d["wildguard"]) > 5e-4:
        raise SystemExit(f"{stem}[{arm}]: reconstructed WildGuard {sum(wg) / len(wg):.4f} != "
                         f"stored {d['wildguard']:.4f}")
    return c, wg


def audited_residuals() -> set[tuple[str, str, int]]:
    """(family, arm, index) of every residual item a human has read and labelled normative."""
    ok = {(x["family"], x["arm"], x["i"]) for x in
          json.load(open(f"{R}/b1_residual_audit.json"))["items"] if x["label"] == "normative"}
    # The 48-token OLMo 2 ablated audit read EVERY WildGuard refusal there (all 80 genuine).
    # A 128-token item inherits that reading iff it was a WildGuard refusal at 48 tokens.
    audit = json.load(open(f"{R}/olmo2_e7_ablated_HANDAUDIT.json"))["arms"]["rlvr_ablated"]
    assert audit["audited"] == audit["n_wildguard_only"] == audit["genuine_refusal"]
    _, w48 = wg_verdicts("olmo2_e7_rlvr", "ablated")
    ok |= {("olmo2_rlvr", "ablated", i) for i, w in enumerate(w48) if w}
    return ok


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, r = p + z * z / (2 * n), z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - r) / d, (c + r) / d


def paired_boot(base: list[bool], abl: list[bool], n: int = 10_000, seed: int = 0) -> tuple:
    """Removed fraction 1 - abl/base, resampling PROMPTS (the same draw in both arms)."""
    b, a = np.array(base, float), np.array(abl, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    bs, as_ = b[idx].sum(1), a[idx].sum(1)
    keep = bs > 0
    r = 1 - as_[keep] / bs[keep]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> None:
    ok_resid = audited_residuals()
    out, mismatches, unaudited = {}, [], []
    for fam, stem in FAMILIES.items():
        arms = {arm: wg_verdicts(stem, arm) for arm in ("baseline", "ablated")}
        row = {"removed_ci_bootstrap_prompts": paired_boot(arms["baseline"][1],
                                                           arms["ablated"][1])}
        for arm, (c, wg) in arms.items():
            st = Counter()
            for i, (x, w) in enumerate(zip(c, wg)):
                if not w:
                    continue
                lab = label(x)
                if lab in ("compliance", "confusion"):
                    if (fam, arm, i) in ok_resid:
                        lab = "normative"
                    else:
                        unaudited.append((fam, arm, i, lab))
                        continue
                st[lab] += 1
            row[arm] = {"wildguard": round(sum(wg) / len(wg), 3), "n_refusals": int(sum(wg)),
                        "stances": {r: st.get(r, 0) for r in REGISTERS}}
        # From COUNTS, not rounded rates (rates gave Tulu-2 70.1 for 30 of 100 surviving).
        nb, na = row["baseline"]["n_refusals"], row["ablated"]["n_refusals"]
        row["removed_pct"] = round(100 * (1 - na / nb), 1)
        # Zero-event-safe interval next to the bootstrap, which is degenerate when the ablated
        # arm has no refusals (every resample then says 100%). Survival treated as na of nb.
        lo_s, hi_s = wilson(na, nb)
        row["removed_ci_wilson_counts"] = (1 - hi_s, 1 - lo_s)
        f = FROZEN[fam]
        for arm, k in (("baseline", 0), ("ablated", 1)):
            if abs(row[arm]["wildguard"] - f[arm]) > 5e-4 or row[arm]["n_refusals"] != f["n"][k]:
                mismatches.append(f"{fam}[{arm}] rate/count {row[arm]['wildguard']}/"
                                  f"{row[arm]['n_refusals']} vs frozen {f[arm]}/{f['n'][k]}")
            if row[arm]["stances"] != f["stances"][arm]:
                mismatches.append(f"{fam}[{arm}] stances {row[arm]['stances']} vs frozen "
                                  f"{f['stances'][arm]}")
        out[fam] = row

    verdict = "REPRODUCED" if not mismatches and not unaudited else "MISMATCH"
    with open(f"{R}/b1_registers_ANALYSIS.json", "w") as fh:
        json.dump({"what": QUESTION, "verdict": verdict, "families": out,
                   "mismatches": mismatches, "unaudited_residuals": unaudited,
                   "n_audited_residuals_used": len(ok_resid)}, fh, indent=1)
    with RunRecord(EXPERIMENT, "b1_registers.py", cfg=None, question=QUESTION,
                   notes="CPU only; regenerates the hand-written B1 table") as rec:
        for fam, row in out.items():
            rec.result(family=fam, removed_pct=row["removed_pct"],
                       baseline=row["baseline"]["wildguard"], ablated=row["ablated"]["wildguard"])
        rec.result(verdict=verdict, mismatches=len(mismatches), unaudited=len(unaudited))

    print(f"\n=== B1: {verdict} ===")
    for fam, row in out.items():
        lo, hi = row["removed_ci_bootstrap_prompts"]
        wl, wh = row["removed_ci_wilson_counts"]
        print(f"{fam:11s} {row['baseline']['wildguard']:.3f} -> {row['ablated']['wildguard']:.3f}"
              f"  removed {row['removed_pct']:5.1f}%  [bootstrap {100 * lo:.0f}, {100 * hi:.0f}]"
              f" [Wilson {100 * wl:.0f}, {100 * wh:.0f}]  | abl {row['ablated']['stances']}")
    for m in mismatches + [f"UNAUDITED {u}" for u in unaudited]:
        print("  !!", m)
    if verdict != "REPRODUCED":
        raise SystemExit("B1 does not reproduce -- see above")


if __name__ == "__main__":
    main()
