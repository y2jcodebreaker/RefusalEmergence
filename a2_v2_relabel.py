"""A2-v2 -- re-score A2's judges with stance_of v2, and measure what v1 got wrong.

    python a2_v2_relabel.py               # CPU only, seconds

WHY. A2's per-register judge table is defined by stance_of v1, which labels a completion by
its OPENING. An answer that opens in a refusal register and then delivers the request
("I do not condone X. However, here are the steps: 1. ...") is filed as a refusal, so a judge
that called it compliance was scored as MISSING a refusal. v2 (stance_v2.py) also asks whether
the text goes on to comply. This script quantifies the difference on A2's frozen input set.

It produces DATA, not a claim. The A2 claim statement and the SUPERSEDED entries that follow
from these numbers are written by hand (P1 claim map, "Pending manual correction").

v1 STAYS FROZEN: judge_bench.py and b1_registers.py must keep reproducing the tables they
define. This script never writes to those outputs.

Output: results/a2_v2_relabel_ANALYSIS.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from a2_gcg import gcg_hits
from judge_bench import A2_FROZEN_N_ARMS, A2_FROZEN_N_ITEMS, REGISTERS, cluster_boot, collect
from stance_directions import stance_of
from stance_v2 import disclaimer_then_comply, stance_of_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("A2-v2")

EXPERIMENT = "A2-v2"
QUESTION = ("How much of A2's apparent WildGuard register bias is v1 labelling "
            "disclaimer-then-comply completions as refusals?")


def main() -> None:
    items, skipped = collect()
    if (len(items), len({(x["file"], x["arm"]) for x in items})) != (A2_FROZEN_N_ITEMS,
                                                                     A2_FROZEN_N_ARMS):
        raise SystemExit("not A2's frozen input set")
    for x in items:
        x["v2"] = stance_of_v2(x["text"])
        x["pivots"] = disclaimer_then_comply(x["text"])

    rng = np.random.default_rng(0)

    # 1. What v1 called a refusal in each register, and how v2 reclassifies it.
    v1_rows = {}
    for reg in REGISTERS:
        rows = [x for x in items if x["register"] == reg]
        if not rows:
            continue
        demoted = [x for x in rows if x["v2"] == "compliance"]
        wg_missed = [x for x in rows if not x["wildguard"]]
        v1_rows[reg] = {
            "n_v1": len(rows),
            "v2_demoted_to_compliance": len(demoted),
            "wildguard_called_compliance": len(wg_missed),
            "of_those_wildguard_missed_v2_demotes": sum(1 for x in wg_missed
                                                        if x["v2"] == "compliance"),
            "families": sorted({x["file"].split("_")[0] for x in rows}),
        }

    # 2. The judges scored on refusals as v2 defines them -- the comparable table.
    v2_rows = {}
    by_v2 = defaultdict(list)
    for x in items:
        by_v2[x["v2"]].append(x)
    for reg in REGISTERS + ("compliance",):
        rows = by_v2.get(reg, [])
        if not rows:
            continue
        cell = {"n": len(rows), "n_arms": len({(r["file"], r["arm"]) for r in rows}),
                "families": sorted({r["file"].split("_")[0] for r in rows})}
        for key, fn in (("jbb12", lambda r: r["substring_verbatim"]),
                        ("gcg", lambda r: bool(gcg_hits(r["text"]))),
                        ("wildguard", lambda r: r["wildguard"])):
            for r in rows:
                r["_k"] = fn(r)
            rate, lo, hi = cluster_boot(rows, "_k", rng)
            cell[key], cell[f"{key}_ci"] = rate, (lo, hi)
        v2_rows[reg] = cell

    # 3. A3's stance classes: how much of each is disclaimer-then-comply?
    a3 = {}
    for stem, arm in (("tulu2_dpo_dpo", "baseline"), ("olmo2_e7_rlvr", "baseline"),
                      ("zephyr_dpo", "baseline")):
        try:
            z = np.load(f"results/{stem}_refusal_gen128.npz", allow_pickle=True)
        except FileNotFoundError:
            continue
        comps = json.loads(str(z["sample_completions"]))[arm]
        per = defaultdict(lambda: [0, 0])
        for c in comps:
            v1 = stance_of(c)
            per[v1][1] += 1
            if stance_of_v2(c) == "compliance" and v1 != "compliance":
                per[v1][0] += 1
        a3[stem] = {k: {"n_v1": v[1], "v2_demotes": v[0]} for k, v in sorted(per.items())}

    out = {"what": QUESTION, "n_items": len(items), "skipped": skipped,
           "v1_registers_and_what_v2_demotes": v1_rows,
           "judges_on_v2_refusals": v2_rows,
           "a3_stance_classes_v1_vs_v2": a3,
           "caveat": "v2 is a heuristic with measured gaps of its own (it misses "
                     "'I strongly disagree' and engage-then-redirect openings). These numbers "
                     "bound the v1 labelling error; they are not gold. Blind human labels "
                     "remain the fix."}
    with open("results/a2_v2_relabel_ANALYSIS.json", "w") as f:
        json.dump(out, f, indent=1, default=float)

    print(f"\n=== A2 under stance_of v2 ({len(items)} items, A2's frozen set) ===")
    print("\n1. Of what v1 called a refusal, how much does v2 demote to compliance?")
    print(f"{'v1 register':14s} {'n':>5s} {'v2 demotes':>11s} {'WG said compliance':>19s} "
          f"{'...of those, demoted':>21s}")
    for reg, r in v1_rows.items():
        print(f"{reg:14s} {r['n_v1']:>5d} {r['v2_demoted_to_compliance']:>11d} "
              f"{r['wildguard_called_compliance']:>19d} "
              f"{r['of_those_wildguard_missed_v2_demotes']:>21d}")
    print("\n2. Judges scored on refusals as v2 defines them:")
    print(f"{'v2 register':14s} {'n':>5s} {'JBB-12':>7s} {'GCG':>7s} {'WildGuard':>10s} "
          f"{'families':>9s}")
    for reg, c in v2_rows.items():
        print(f"{reg:14s} {c['n']:>5d} {c['jbb12']:>7.3f} {c['gcg']:>7.3f} "
              f"{c['wildguard']:>10.3f} {len(c['families']):>9d}")
    print("\n3. A3's stance classes (baseline arms), v1 count -> how many v2 demotes:")
    for stem, d in a3.items():
        print(f"  {stem:16s} " + "  ".join(f"{k} {v['n_v1']}->{v['n_v1'] - v['v2_demotes']}"
                                            for k, v in d.items() if v["n_v1"]))
    print("\n  -> results/a2_v2_relabel_ANALYSIS.json")


if __name__ == "__main__":
    main()
