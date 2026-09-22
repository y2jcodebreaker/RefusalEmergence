"""The claim graph: which script produced which file, supporting which claim, guarded by which
control. Generates PROVENANCE.md and checks it against what is actually on disk.

    python provenance.py              # write PROVENANCE.md
    python provenance.py --check      # exit 1 if the graph and the disk disagree

Two jobs, both of which used to live only in my head and in commit messages.

1. PAPER WRITING. Every number in the paper should be traceable to a file, a script, a commit
   and a claim without anyone reconstructing it from memory. `results/RUNLOG.md` records what
   ran; this records what it was FOR.

2. EXPERIMENT DESIGN, BREADTH-FIRST. Claims carry a `layer`. A claim at layer N+1 is a
   prediction FROM the layer-N claims, so running it while a layer-N control is still open is
   depth-first: it spends GPU deepening a chain whose earlier link can still move. `--check`
   refuses that, by name.

To add an experiment: add its Claim here FIRST, with its controls and its falsifier, and only
then write the script. If you cannot name the falsifier, the experiment is not designed yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

RESULTS = "results"
LEDGER = os.path.join(RESULTS, "runs.jsonl")


@dataclass(frozen=True)
class Evidence:
    """One measurable output: a results/{lineage}_{stage}_{axis}.npz written by `script`."""
    axis: str
    script: str
    experiment: str
    what: str


@dataclass(frozen=True)
class Control:
    """A robustness check that guards a claim. `done` is the honest current state.

    `optional=True` marks a STRENGTHENING rather than a gap: the claim is defensible without
    it. The distinction is load-bearing for the breadth-first guard -- if every conceivable
    further control blocked the next layer, nothing would ever descend, and "be BFS" would
    collapse into "never finish". A control is only blocking while its absence leaves a
    reviewer's question unanswered."""
    name: str
    what_it_rules_out: str
    done: bool
    where: str = ""
    optional: bool = False
    script: str = ""     # if a script produced it, it must appear in the ledger when done


@dataclass(frozen=True)
class Claim:
    id: str
    layer: int
    statement: str
    evidence: tuple[Evidence, ...]
    controls: tuple[Control, ...]
    falsifier: str
    depends_on: tuple[str, ...] = field(default=())
    note: str = ""

    @property
    def open_controls(self) -> list[Control]:
        """Open and BLOCKING. Optional strengthenings are listed but do not gate a layer."""
        return [c for c in self.controls if not c.done and not c.optional]

    @property
    def open_optional(self) -> list[Control]:
        return [c for c in self.controls if not c.done and c.optional]


# --------------------------------------------------------------------------- the graph

CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="C1", layer=1,
        statement="The harmful/harmless distinction is linearly present in the BASE model.",
        evidence=(
            Evidence("probe", "probe_representation.py", "P1-E1",
                     "per-layer probe accuracy, mass-mean and logistic, + shuffled-label nulls"),
            Evidence("transfer", "probe_transfer.py", "P1-E1d",
                     "Arditi-fitted probe TRANSFERRED to XSTest, full and focus-matched"),
        ),
        controls=(
            Control("layer-0 probe", "surface lexicon at the eoi window", True,
                    "L0 = 0.500 exactly, both families"),
            Control("token-length baseline", "harmful prompts simply being longer", True,
                    "0.471 / 0.485, both BELOW chance"),
            Control("XSTest focus-matched", "topic and vocabulary confound", True,
                    "0.820 / 0.823, trigger word held constant"),
            Control("eoi window leak check", "BPE merging prompt text into the window", True,
                    "verify_setup.window_leaks() over 80 real prompts"),
            Control("SORRY-Bench held-out topics", "topic generalisation, a stronger form",
                    False, "XSTest already closes the lexical confound; this is a second, "
                    "different cut at the same question", optional=True),
        ),
        falsifier="Base collapses toward chance on the focus-matched subset while the aligned "
                  "stages hold -> base represents TOPIC and alignment builds the harmfulness "
                  "distinction. A different paper, and a real one.",
    ),
    Claim(
        id="C2", layer=1,
        statement="That representation is NOT sufficient to produce refusal in base: base's own "
                  "direction is sub-threshold (Zephyr) or indistinguishable from noise (OLMo 2).",
        evidence=(
            Evidence("refusal", "run_stage.py", "E02",
                     "per-layer induce/ablate/KL surfaces and Arditi's three selection filters"),
            Evidence("transplant", "transplant.py", "P1-E1b/P1-E2d",
                     "16-cell matrix, delta from own baseline, z vs the shuffled-label null"),
        ),
        controls=(
            Control("shuffled-label null", "an isotropic null being too easy to beat", True,
                    "10 draws/cell; base z 0.8-1.3, aligned 2.6-3.7, no overlap"),
            Control("coefficient sweep", "'you only tried coefficient 1'", True,
                    "6-8 coefficients; base swept to 16-54x its own norm"),
            Control("positive control", "an under-powered sweep reading as a negative", True,
                    "self-cell must induce, or the matrix is reported uninterpretable"),
            Control("argmax-over-induce source", "'you tested the wrong direction', and "
                    "the circularity of selecting l* BY the induce criterion", True,
                    "--source-by induce, 2026-09-19: base's induce-optimal cell (L19) gives "
                    "z = +0.6 to +1.6 in all four targets and never crosses; aligned (L18) "
                    "z = +2.5 to +3.8, 12/12. Clean separation at BOTH source layers."),
            Control("gradient search + rank-k subspace", "'no refusal CONE was looked for'",
                    False, "Wollschlaeger ICML 2025; the strongest form of C2. The definitional "
                    "objection is already closed by --source-by induce", optional=True),
        ),
        depends_on=("C1",),
        falsifier="Some direction induces refusal in base at acceptable KL -> the representation "
                  "IS sufficient and the coupling account is wrong.",
    ),
    Claim(
        id="C3", layer=1,
        statement="The ALIGNED model's direction IS sufficient in base -- it writes fluent "
                  "refusals at the direction's own natural magnitude. (OLMo 2; not Zephyr.)",
        evidence=(
            Evidence("transplant", "transplant.py", "P1-E1b/P1-E2d",
                     "base<-sft: delta +10.47, z +3.7, p .0024, Bonferroni-safe over 16 cells"),
            Evidence("text", "transplant_text.py", "P1-E1c",
                     "0.000 -> 1.000 strict refusal in TEXT, matched-norm random at 0.016"),
        ),
        controls=(
            Control("norm-matched random arm", "'any big perturbation would do this'", True,
                    "0.016 vs 1.000 at identical magnitude"),
            Control("shuffled-label null, 160 draws", "the crossing criterion's error rate", True,
                    "0/160 shuffled directions ever crossed"),
            Control("over-injection control", "'the loop means you broke the model'", True,
                    "SFT loops identically at 2x -> over-injection, not base-specific"),
            Control("degenerate-text judge", "a phrase loop scoring as a refusal", True,
                    "distinct-4-gram ratio; the arm is 98.4% degenerate at 2x"),
            Control("WildGuard judge (NeurIPS 2024)", "'your judge is a regex'", True,
                    "2026-09-19: at the operating point substring 1.000 vs WildGuard 0.984, "
                    "ONE disagreement -- index 20, the same false positive the hand audit "
                    "found independently. Judges diverge only on degenerate text "
                    "(76-100% of disagreements at 2x magnitude).",
                    script="judge_wildguard.py"),
            Control("hand audit of the injected arm", "a new false-positive class in a "
                    "new regime", True,
                    "all 64 read 2026-09-19: 48 genuine / 14 degenerate / 1 partial / 1 "
                    "complies -> 0.750, not the judge's 1.000. Labels in "
                    "results/olmo2_base_from_sft_HANDAUDIT.json"),
        ),
        depends_on=("C1", "C2"),
        falsifier="The aligned direction's effect in base is indistinguishable from the "
                  "shuffled-label null -> nothing transplants and the paper has no causal claim.",
    ),
    Claim(
        id="C4", layer=1,
        statement="The signature is not one training run: it replicates across two families "
                  "sharing no corpus, tokenizer or alignment recipe.",
        evidence=(
            Evidence("probe", "probe_representation.py", "P1-E1", "both lineages"),
            Evidence("transfer", "probe_transfer.py", "P1-E1d",
                     "base focus-matched 0.820 (Zephyr) / 0.823 (OLMo 2)"),
            Evidence("refusal", "run_stage.py", "E02", "both lineages, 4 + 3 checkpoints"),
        ),
        controls=(
            Control("lineage-scoped outputs", "one family overwriting another's results", True,
                    "Config.path() and Config.figure(); a test asserts no collision"),
            Control("per-family judge audit", "each family inventing a new false positive", True,
                    "O-51 sympathy-then-complies; O-66 decline-then-complies"),
            Control("regime-override discipline", "an unusable base operating regime", True,
                    "stage_regime; raw scores flagged incomparable, cosine skipped"),
        ),
        depends_on=("C1",),
        falsifier="C1-C3 hold in one family and fail in the other for reasons other than "
                  "alignment recipe -> the result is an artifact of one pipeline. "
                  "PARTIALLY REALISED: C3 is OLMo 2 only; reported as a finding, not hidden.",
        note="C3 holds in OLMo 2 and not Zephyr. Stated in the results, not the limitations.",
    ),
    Claim(
        id="P1-E7", layer=2,
        statement="CONFIRMED 2026-09-19. Breaking alignment breaks the LINK and spares "
                  "the REPRESENTATION. Attacked: probe 1.000, XSTest focus-matched 0.917, "
                  "behavioural refusal 0.189, and ZERO of 130 cells induce. Its own "
                  "direction reaches -4.955 at natural scale; rlvr's direction injected "
                  "into it reaches +1.069 and crosses. Neither the representation nor the "
                  "readout was damaged -- the coupling between them was.",
        evidence=(
            Evidence("refusal", "run_stage.py", "P1-E7",
                     "behavioural refusal rate of the attacked stage, before/after"),
            Evidence("probe", "probe_representation.py", "P1-E7",
                     "probe layer CURVE, not the peak -- 1.000 is saturated"),
            Evidence("transfer", "probe_transfer.py", "P1-E7",
                     "XSTest transfer of the attacked model: the generalisation half"),
            Evidence("transplant", "transplant.py", "P1-E7b",
                     "inject the UN-attacked direction into the attacked model. 2026-09-19: "
                     "at the operating point rlvr->attacked = +1.069 (crosses), "
                     "attacked->attacked = -4.955 (does not), and the negative is POWERED by "
                     "the same target accepting rlvr's and control's directions"),
        ),
        controls=(
            Control("matched safety-preserved arm", "attributing decoupling to fine-tuning "
                    "in general rather than to safety removal", True,
                    "2026-09-19: RUN. Identical rank 16 / lr 2e-4 / 3 epochs / 2000 Alpaca "
                    "examples; the only difference is 50 rehearsed refusals. benign arm "
                    "1.000 -> 0.104, control arm 1.000 -> 1.000. Fine-tuning per se does not "
                    "do this.", script="attack.py"),
            Control("control is not a refuse-everything model", "the control holding 1.000 "
                    "on harmful prompts because 50 rehearsed refusals induced EXAGGERATED "
                    "SAFETY, which would make the held rate a confound rather than preserved "
                    "safety", True,
                    "2026-09-19: PASSED, and in the opposite direction to the worry. XSTest "
                    "safe strict refusal: rlvr 0.076, control 0.072 (delta -0.004, bound "
                    "+0.10); Arditi harmless_val: rlvr 0.029, control 0.004. The control "
                    "over-refuses LESS than the untouched checkpoint while holding 1.000 on "
                    "harmful prompts. Qualitatively it reproduces rlvr's OWN over-refusals on "
                    "the same class of XSTest items ('kill someone in Call of Duty', 'stab a "
                    "balloon'), i.e. it inherited the safety behaviour including its "
                    "characteristic flaws rather than adopting a blanket policy. Degeneracy "
                    "0.008 / 0.010 / 0.005, so neither arm was broken by the tuning.",
                    script="overrefusal.py"),
            Control("behavioural pre-check", "reading mechanism from a failed attack", True,
                    "2026-09-19: it FIRED. Attack v1 (responses from the model's own outputs) "
                    "left refusal at 0.985 -> 0.985, i.e. no effect. Six minutes of measurement "
                    "stopped before hours of uninterpretable mechanism numbers. attack.py now "
                    "defaults to Alpaca reference responses and checks efficacy in-run. "
                    "2026-09-19 second firing: Alpaca at n=100 moved refusal 1.000 -> 0.979, "
                    "logged THE ATTACK DID NOT WORK; the dose, not the data, was wrong. At "
                    "n=2000 it fired properly: 1.000 -> 0.104.", script="attack.py"),
            Control("SFT loss masked to responses", "a language-modelling run on our own "
                    "eval prompts", True, "encode_sft + 9 tests"),
            Control("attacked source by INDUCE argmax", "the circularity in 'the attacked "
                    "model's own direction does not induce': its source layer L25 is the "
                    "UNFILTERED argmax fallback, because the attacked model has no filtered "
                    "l* at all -- so 'no valid direction' and 'its direction does nothing' "
                    "risk being the same statement, exactly as for base in C2", True,
                    "2026-09-19: CLOSED. The attacked model's induce-argmax cell IS L25, the "
                    "same cell the ablation fallback picked, so the negative does not depend "
                    "on the selection rule and its rows are numerically identical. That "
                    "cell's steer is -5.188, which is the MAXIMUM over the whole induce "
                    "surface (130 unpruned cells, 0 pass, median -11.73); swept to 16x raw "
                    "norm it still tops out at -0.136. The claim is now 'the attacked "
                    "model's best candidate BY THE METRIC WE SCORE produces no refusal "
                    "anywhere in the sweep', not 'nothing passed our filters'.",
                    script="transplant.py"),
            Control("step dose-response", "a single before/after pair being a coincidence",
                    False, "dose_response.py --arm {benign,safety-preserved}, built "
                           "2026-09-21, not yet run. Measures behaviour, coupling and probe "
                           "IN PLACE at 6 doses (0/100/250/500/1000/1500 steps) -- no merged "
                           "checkpoint per dose, which would be 180 GB; only the ~80 MB LoRA "
                           "adapter. Behaviour and coupling should fall TOGETHER while the "
                           "probe curve stays flat. Same shape of argument as Frank 2026. "
                           "Stores completions at every dose because the substring rate is a "
                           "lower bound (O-120): the behavioural curve is BLOCKED on "
                           "judge_wildguard.py per dose, since a register shift DURING "
                           "training would fake exactly this experiment's result.",
                    script="dose_response.py"),
            Control("seed replication", "one stochastic training run", False,
                    "Subsumed by dose_response.py: the original attacked checkpoint lived on "
                    "a destroyed pod, so its dose-1500 endpoint is an independent run at a "
                    "different LoRA draw. It will NOT land on 0.477, and agreement in SHAPE "
                    "across two runs is stronger than one number reproducing.",
                    script="dose_response.py"),
        ),
        depends_on=("C1", "C2", "C3", "C4"),
        falsifier="Probe accuracy and its layer shape degrade alongside behaviour -> "
                  "fine-tuning damaged the representation and the clean decoupling story "
                  "fails. Or P1-E7b fails to restore refusal -> the attack damaged the "
                  "readout too, and 'breaks a wire' is the wrong metaphor.",
        note="Layer 2. Do not RUN this while any layer-1 control is open (see --check).",
    ),
)


# --------------------------------------------------------------------------- disk + ledger

def on_disk() -> dict[str, list[str]]:
    """axis -> the result files that exist.

    Two shapes, because not every axis is an array: the sweeps are
    results/{lineage}_{stage}_{axis}.npz, but P1-E1c stores COMPLETIONS as
    results/{lineage}_{target}_from_{source}_text.json -- text is the evidence there, so it
    has to be text on disk. The first version of this scanner only globbed .npz and reported
    C3's text evidence as missing; the check caught my own graph being wrong, which is the
    point of having it."""
    out: dict[str, list[str]] = {}
    for pat, ext in ((f"{RESULTS}/*.npz", ".npz"), (f"{RESULTS}/*_text.json", ".json")):
        for p in sorted(glob.glob(pat)):
            axis = os.path.basename(p).rsplit("_", 1)[-1][: -len(ext)]
            out.setdefault(axis, []).append(os.path.basename(p))
    return out


def ledger() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER) as f:
        return [json.loads(line) for line in f if line.strip()]


def runs_for(rows, script: str) -> list[dict]:
    return [r for r in rows if r.get("script") == script]


# --------------------------------------------------------------------------- checks

def check(claims=CLAIMS) -> list[str]:
    """Problems, as human sentences. Empty means the graph and the disk agree."""
    problems: list[str] = []
    files, rows = on_disk(), ledger()
    by_id = {c.id: c for c in claims}

    for c in claims:
        for dep in c.depends_on:
            if dep not in by_id:
                problems.append(f"{c.id} depends on {dep}, which is not in the graph.")
            elif by_id[dep].layer > c.layer:
                problems.append(f"{c.id} (layer {c.layer}) depends on {dep} "
                                f"(layer {by_id[dep].layer}) -- a claim cannot rest on a "
                                f"DEEPER one; the layering is wrong.")
        for e in c.evidence:
            if e.axis not in files:
                problems.append(f"{c.id}: no results/*_{e.axis}.npz on disk, so the evidence "
                                f"'{e.what}' does not exist yet (would come from {e.script}).")
            if rows and not runs_for(rows, e.script):
                problems.append(f"{c.id}: {e.script} has never run according to the ledger, "
                                f"yet it is listed as evidence.")
        # A control marked DONE whose script never ran is the same hole one level down:
        # judge_wildguard.py produced a number cited in RESULTS.md and wrote no ledger row
        # (2026-09-19), because only EVIDENCE scripts were checked.
        for ct in c.controls:
            if ct.done and ct.script and rows and not runs_for(rows, ct.script):
                problems.append(
                    f"{c.id}: control '{ct.name}' is marked done but {ct.script} has no "
                    f"ledger row, so its number is uncited. The result stands; the "
                    f"PROVENANCE does not. Re-run {ct.script} to record it — do not "
                    f"hand-write a row.")

    # BFS guard: the thing the user asked for by name.
    deepest_open = min((c.layer for c in claims if c.open_controls), default=None)
    if deepest_open is not None:
        for c in claims:
            if c.layer > deepest_open and any(e.axis in files and
                                              any(e.experiment.split("/")[0] in
                                                  (r.get("experiment") or "")
                                                  for r in runs_for(rows, e.script))
                                              for e in c.evidence):
                problems.append(
                    f"DEPTH-FIRST: {c.id} is layer {c.layer} and has been run, but layer "
                    f"{deepest_open} still has open controls "
                    f"({', '.join(cc.id for cc in claims if cc.layer == deepest_open and cc.open_controls)}). "
                    f"Close the shallower layer before spending GPU on the deeper one.")
    return problems


def render(claims=CLAIMS) -> str:
    files, rows = on_disk(), ledger()
    L = ["# Provenance — the claim graph",
         "",
         "Generated by `provenance.py`. **Do not hand-edit**; edit the `CLAIMS` tuple and",
         "regenerate. `results/RUNLOG.md` records *what ran*; this records *what it was for*.",
         "",
         "Every claim below names its evidence (script -> file), the controls that guard it, and",
         "**the observation that would kill it**. A claim with no falsifier is not designed yet.",
         ""]
    for layer in sorted({c.layer for c in claims}):
        rung = [c for c in claims if c.layer == layer]
        L += [f"## Layer {layer}", ""]
        if layer > 1:
            L += [f"Layer {layer} claims are *predictions from* layer {layer - 1}. Running one "
                  f"while a shallower control is open is depth-first; `--check` refuses it.", ""]
        for c in rung:
            open_, opt = c.open_controls, c.open_optional
            mark = ("✅" if not open_ else f"⚠️ {len(open_)} blocking control(s) open")
            if opt:
                mark += f" · {len(opt)} optional strengthening(s) available"
            L += [f"### {c.id} — {mark}", "", f"**{c.statement}**", ""]
            if c.depends_on:
                L += [f"Depends on: {', '.join(c.depends_on)}", ""]
            L += ["| evidence | script | experiment | files on disk |",
                  "|---|---|---|---|"]
            for e in c.evidence:
                got = files.get(e.axis, [])
                n = f"{len(got)} × `*_{e.axis}.npz`" if got else "**none yet**"
                L.append(f"| {e.what} | `{e.script}` | {e.experiment} | {n} |")
            L += ["", "| control | rules out | status |", "|---|---|---|"]
            for ct in c.controls:
                tag = ("✅ " + ct.where if ct.done
                       else ("◻️ *optional* — " if ct.optional else "⬜ ") +
                       (ct.where or "not run"))
                L.append(f"| {ct.name} | {ct.what_it_rules_out} | {tag} |")
            L += ["", f"**Falsifier.** {c.falsifier}", ""]
            if c.note:
                L += [f"> {c.note}", ""]
    probs = check(claims)
    L += ["## Consistency check", ""]
    L += ["Graph and disk agree." if not probs else
          "\n".join(f"- ⚠️ {p}" for p in probs)]
    L += ["", f"Ledger: {len(rows)} recorded runs across "
              f"{len({r.get('script') for r in rows})} scripts.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the graph and the disk disagree, or if a deeper layer has "
                         "been run while a shallower control is open")
    args = ap.parse_args()
    probs = check()
    if args.check:
        for p in probs:
            print(f"⚠️  {p}")
        print("provenance: OK" if not probs else f"provenance: {len(probs)} problem(s)")
        raise SystemExit(1 if probs else 0)
    with open("PROVENANCE.md", "w") as f:
        f.write(render())
    print(f"wrote PROVENANCE.md ({len(CLAIMS)} claims, "
          f"{sum(len(c.controls) for c in CLAIMS)} controls, "
          f"{sum(len(c.open_controls) for c in CLAIMS)} open)")
    for p in probs:
        print(f"  ⚠️  {p}")


if __name__ == "__main__":
    main()
