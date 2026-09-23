"""P1-E7 correction — the matched control, re-scored on prompts it never trained on.

    python heldout_control.py            # CPU only, reads stored completions + verdicts

WHAT WENT WRONG. The safety-preserved control rehearses its own refusals to harmful prompts.
Until 2026-09-23 those rehearsal prompts were drawn from harmful_train[n_train:] -- the same
132-prompt tail every behavioural measurement scores -- and attack.py's efficacy check used
tail[:48], which lies entirely inside the 50 rehearsed prompts. So the control's quoted
efficacy figure was a memorisation readout, and every control-arm behavioural rate mixed 50
trained-on prompts with 82 unseen ones.

WHAT THIS SCRIPT ESTABLISHES, and why each step is needed:

  1. WHICH prompts were rehearsed -- reconstructed, not assumed. build_safety_examples keeps
     the first 50 strict refusals of the untouched model's greedy completions. The control
     run's dose-0 completions ARE those completions, and are asserted identical to
     run_stage's rlvr file before anything is read from them.
  2. Per-item WildGuard verdicts, recovered exactly from the stored disagreement indices and
     asserted against the stored rate (the same identity A2 uses).
  3. A DIFFERENCE-IN-DIFFERENCES, not a raw split. The rehearsed prompts are tail[0:50], and
     the ATTACK arm -- which never saw them -- also scores higher on them, so part of any
     rehearsed-vs-held-out gap is prompt difficulty. The control's advantage over the attack
     is therefore computed WITHIN each prompt set. If contamination were manufacturing the
     preservation effect, the advantage would be larger on rehearsed prompts; it is not.

Output: results/p1e7_heldout_ANALYSIS.json
"""

from __future__ import annotations

import json

import numpy as np

from refusal_substring import is_refusal_strict, truncate_at_turn
from runlog import RunRecord

EXPERIMENT = "P1-E7-heldout"
QUESTION = ("Does the P1-E7 matched control still preserve refusal when scored only on "
            "prompts it did not rehearse?")
R = "results"
REHEARSAL_N = 50
DOSES = (100, 250, 500, 1000, 1500)


def comps(path: str, arm: str = "baseline") -> list[str]:
    return json.loads(str(np.load(path, allow_pickle=True)["sample_completions"]))[arm]


def wildguard(stem: str, completions: list[str]) -> tuple[list[bool], list[bool]]:
    """(wildguard, substring) per item, reconstructed and asserted against the stored rate."""
    sub = [is_refusal_strict(truncate_at_turn(c)) for c in completions]
    rep = json.load(open(f"{R}/{stem}_wildguard.json"))["baseline"]
    dis = set(rep.get("disagreements") or [])
    wg = [(not sub[i]) if i in dis else sub[i] for i in range(len(completions))]
    if abs(sum(wg) / len(wg) - rep["wildguard"]) > 5e-4:
        raise SystemExit(f"{stem}: reconstructed WildGuard {sum(wg) / len(wg):.4f} != stored "
                         f"{rep['wildguard']:.4f} -- these are not the completions judged")
    return wg, sub


def rate(v: list[bool], idx: list[int]) -> float:
    return sum(v[i] for i in idx) / len(idx)


def main() -> None:
    d0 = comps(f"{R}/olmo2_e7d_safety-preserved_dose_0_refusal.npz")
    rl = comps(f"{R}/olmo2_e7_rlvr_refusal.npz")
    if d0 != rl:
        raise SystemExit("the control run's dose-0 completions differ from run_stage's rlvr "
                         "file, so they are not the completions the rehearsal set was chosen "
                         "from -- the reconstruction would be of the wrong set")
    kept = [i for i, c in enumerate(d0) if is_refusal_strict(c)][:REHEARSAL_N]
    held = [i for i in range(len(d0)) if i not in set(kept)]

    did = []
    for d in DOSES:
        c, _ = wildguard(f"olmo2_e7d_safety-preserved_dose_{d}",
                         comps(f"{R}/olmo2_e7d_safety-preserved_dose_{d}_refusal.npz"))
        a, _ = wildguard(f"olmo2_e7d_benign_dose_{d}",
                         comps(f"{R}/olmo2_e7d_benign_dose_{d}_refusal.npz"))
        did.append({"dose": d,
                    "control_rehearsed": rate(c, kept), "control_heldout": rate(c, held),
                    "attack_rehearsed": rate(a, kept), "attack_heldout": rate(a, held),
                    "advantage_rehearsed": rate(c, kept) - rate(a, kept),
                    "advantage_heldout": rate(c, held) - rate(a, held)})

    endpoints = {}
    for name, stem in (("control", "olmo2_e7_control"), ("attacked", "olmo2_e7_attacked"),
                       ("rlvr", "olmo2_e7_rlvr")):
        w, s = wildguard(stem, comps(f"{R}/{stem}_refusal.npz"))
        endpoints[name] = {"wildguard_all": rate(w, list(range(len(w)))),
                           "wildguard_heldout": rate(w, held),
                           "substring_all": rate(s, list(range(len(s)))),
                           "substring_heldout": rate(s, held)}
    gap_held = endpoints["control"]["wildguard_heldout"] - endpoints["attacked"]["wildguard_heldout"]
    gap_all = endpoints["control"]["wildguard_all"] - endpoints["attacked"]["wildguard_all"]
    survives = all(r["advantage_heldout"] >= r["advantage_rehearsed"] - 1e-9 for r in did)

    out = {"rehearsed_indices": kept, "n_rehearsed": len(kept), "n_heldout": len(held),
           "per_dose": did, "endpoints": endpoints,
           "endpoint_gap_heldout": gap_held, "endpoint_gap_all": gap_all,
           "advantage_not_larger_on_rehearsed_at_any_dose": survives,
           "generation_tokens": 48}
    path = f"{R}/p1e7_heldout_ANALYSIS.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    with RunRecord(EXPERIMENT, "heldout_control.py", cfg=None, question=QUESTION,
                   notes=f"rehearsed {len(kept)} / held-out {len(held)}; WildGuard verdicts "
                         f"reconstructed and asserted; 48-token completions") as rec:
        for r in did:
            rec.result(dose=r["dose"], advantage_rehearsed=round(r["advantage_rehearsed"], 4),
                       advantage_heldout=round(r["advantage_heldout"], 4))
        rec.result(dose="endpoint", gap_heldout=round(gap_held, 4), gap_all=round(gap_all, 4),
                   conclusion_survives=survives)

    print(f"\n=== P1-E7 control, re-scored where it did not train "
          f"({len(held)} held-out of {len(d0)}) ===\n")
    print(f"{'dose':>5} {'ctrl-atk rehearsed':>19} {'ctrl-atk HELD-OUT':>18}")
    for r in did:
        print(f"{r['dose']:>5} {r['advantage_rehearsed']:>+19.3f} {r['advantage_heldout']:>+18.3f}")
    print("\nendpoints (WildGuard, 48 tok):")
    for k, v in endpoints.items():
        print(f"  {k:9s} all {v['wildguard_all']:.3f}   held-out {v['wildguard_heldout']:.3f}")
    print(f"\n  control - attacked gap: {gap_held:.3f} held-out vs {gap_all:.3f} over all")
    print(f"  advantage never larger on rehearsed prompts: {'YES' if survives else 'NO'}")
    print(f"  -> {'the conclusion survives; restate the numbers on held-out prompts' if survives else 'CONTAMINATION INFLATES THE CONTROL EFFECT -- the P1-E7 control claim needs rework'}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
