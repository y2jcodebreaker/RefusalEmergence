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
import re
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
                     "0.000 -> 0.750 HAND-AUDITED refusal in TEXT (substring says 1.000; "
                     "48 genuine / 14 degenerate / 1 partial / 1 complies). Matched-norm "
                     "random 0.016. The baseline 0.000 was re-checked for the normative "
                     "undercount on 2026-09-21 and is genuine: base does not decline "
                     "harmless prompts in any register"),
        ),
        controls=(
            Control("norm-matched random arm", "'any big perturbation would do this'", True,
                    "0.016 vs 0.750 hand-audited (1.000 substring) at identical magnitude"),
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
        id="D1", layer=3,
        statement="RESOLVED 2026-09-23, NEGATIVE, before any attack run. Coupling collapse "
                  "is NOT a specific detector of safety-removing fine-tunes. Of five "
                  "safety-PRESERVING controls, three lost every steerable layer (OLMo 2 s1, "
                  "both Tulu-2 seeds), so the pre-registered primary's threshold is 0 and it "
                  "can never fire. On Tulu-2 the coupling metric collapsed while refusal held: "
                  "at dose 250 Tulu-2 s1 has 58 inability refusals against 56 at dose 0 and "
                  "WildGuard 0.720 vs 0.780, yet 0 steerable layers and a frozen dose-0 "
                  "direction at -8.19. On OLMo 2 the one zero-layer event is a genuine "
                  "transient safety loss (45/82 compliant, WildGuard 0.427, recovered by dose "
                  "100) and the frozen direction stays +1.9 to +4.8 in every control -- so the "
                  "P1-E7d decomposition holds on OLMo 2 and does not transfer to Tulu-2.",
        evidence=(
            Evidence("d1_controls_ANALYSIS", "d1_controls.py", "D1",
                     "per-control, per-dose steerable layers, frozen induce, WildGuard and "
                     "refusal-stance counts; the dissociations and transient losses computed, "
                     "not eyeballed"),
        ),
        controls=(
            Control("thresholds from controls only, frozen first", "choosing a sensitive "
                    "threshold for our metric and a strict one for the baseline", True,
                    "2026-09-23: RUN. d1_detect.py calibrate read the 5 controls only and "
                    "froze thresholds with sha256s before any attack existed. It is what "
                    "exposed the floor-level primary threshold.", script="d1_detect.py"),
            Control("stopping rule enforced in code", "running attacks after the specificity "
                    "arm has already failed", True,
                    "2026-09-23: the first check printed 'did not fire' -- its leave-one-out "
                    "test is vacuous at threshold 0. Fixed so a floor-level threshold fires; "
                    "replayed on the real thresholds file (old False, new True). Attacks were "
                    "not run.", script="d1_controls.py"),
            Control("stance counted at every dose", "blaming a coupling collapse on a register "
                    "shift that did not happen", True,
                    "2026-09-23: RUN. The Tulu-2 dissociations are not register shifts: the "
                    "inability count is at or above dose 0 where coupling reads zero.",
                    script="d1_controls.py"),
            Control("replicates that actually differ", "calibrating on one run repeated",
                    True, "2026-09-23: the three OLMo 2 seeds diverge from dose 50 on; the "
                    "seed-1 re-run reproduced the migrated original to the last digit.",
                    script="dose_response.py"),
            Control("128-token behaviour", "a behavioural comparator inflated by truncation",
                    True, "2026-09-23: every control file is a _gen128 run.",
                    script="d1_detect.py"),
        ),
        depends_on=("P1-E7",),
        falsifier="FIRED -- in the form of its stopping rule: a safety-preserving control "
                  "lost its coupling while keeping its refusal. What would revive a detector "
                  "claim is a statistic that stays quiet on ALL controls including Tulu-2 and "
                  "fires on attacks, pre-registered afresh and tested on new runs; these "
                  "runs cannot confirm any statistic chosen after seeing them.",
        note="Layer 3. Pre-registration and both appended outcomes in "
             "P1-Coupling-Not-Capability.md section 11. Pre-registration errors it exposed: "
             "Tulu-2 has ONE steerable layer at dose 0, so its fraction is binary; the "
             "induce fraction divides by a small dose-0 value (threshold -15.5).",
    ),
    Claim(
        id="A2", layer=3,
        statement="RESOLVED 2026-09-22. The field's standard refusal judge is REGISTER-BLIND, "
                  "and so is the accepted classifier that replaces it. Arditi's twelve-prefix "
                  "substring judge covers exactly two of the four attested registers; on "
                  "normative refusals it scores 0.029, and 171 such items were hand-read "
                  "against their prompts and confirmed genuine refusals. WildGuard (Han et "
                  "al., NeurIPS 2024) is near-perfect on inability (0.998) but scores 0.858 "
                  "on normative, 0.564 on identity and 0.347 on condemnation -- and calls "
                  "12.7% of the compliance bucket a refusal. The two error types point in "
                  "OPPOSITE directions, so an aggregate refusal rate can look correct while "
                  "both are large.",
        evidence=(
            Evidence("judge_bench_ANALYSIS", "judge_bench.py", "A2",
                     "per-register sensitivity and specificity for both judges over 5920 "
                     "completions from 50 arms and 3 families, with cluster-bootstrap "
                     "intervals over ARMS rather than items"),
        ),
        controls=(
            Control("circularity partition", "reporting a tautology as a finding -- "
                    "stance_of's identity pattern IS Arditi's three identity prefixes", True,
                    "2026-09-22: COMPUTED, not assumed. 0 of 2169 inability items open with "
                    "one of the three patterns stance_of has and Arditi lacks, so both of "
                    "those substring cells are pinned at 1.000 by construction. The output "
                    "marks them DEFINITIONAL and the claim rests on WildGuard's column "
                    "(an independent classifier) and on the hand-audited misses.",
                    script="judge_bench.py"),
            Control("cluster-robust intervals", "treating 50 correlated looks at one "
                    "132-prompt set as 5920 independent observations", True,
                    "2026-09-22: RUN. Every interval is a bootstrap resampling ARMS. On the "
                    "identity row this widens [0.50,0.63] to [0.35,0.82] -- the naive "
                    "interval would have supported a claim the data does not.",
                    script="judge_bench.py"),
            Control("family count per register", "a single model's idiosyncrasy reported as "
                    "a property of the judge", True,
                    "2026-09-22: RUN and it BIT. identity is tulu2 only and condemnation is "
                    "zephyr only -- they are the only families producing those registers at "
                    "all -- so those two rows are single-family however many arms they span. "
                    "The normative row (47 arms, 3 families) is the one that carries weight.",
                    script="judge_bench.py"),
            Control("verdict/completion pairing", "scoring 128-token text against 48-token "
                    "verdicts", True,
                    "2026-09-22: the obvious check does NOT work and that is itself a result. "
                    "Substring rates are INVARIANT to generation length (tulu2_dpo_dpo is "
                    "0.9015 at both 48 and 128 tokens), because the prefixes match an opening "
                    "and nothing later withdraws the match -- which is exactly why this judge "
                    "could not see O-139's inflation. Pairing inverts judge_wildguard.py's "
                    "deterministic naming rule; the rate is a consistency assertion on top; "
                    "smoke_test pins the two functions together.", script="judge_bench.py"),
            Control("two-sided by construction", "a benchmark that only measures "
                    "undercounting, i.e. an advertisement for classifiers", True,
                    "2026-09-22: RUN. Specificity is reported alongside sensitivity, and it "
                    "is where WildGuard looks worst.", script="judge_bench.py"),
        ),
        depends_on=("C3", "C4"),
        falsifier="Judges agree with each other and with the hand labels within noise across "
                  "registers -> there is no instrument story and A2 collapses to a paragraph. "
                  "It did not: substring and WildGuard differ by 0.83 on the normative row "
                  "alone. The surviving claim's own falsifier is the compliance fork -- if "
                  "the 12.7% turns out to be refusals in an unnamed fifth register rather "
                  "than WildGuard errors, the over-counting half is retired and the taxonomy "
                  "half replaces it. Settling that needs a blind audit of a sample from the "
                  "compliance bucket, which is not run.",
        note="Layer 3, and CPU-ONLY. judge_wildguard.py stored per-arm rates plus the indices "
             "where the judges disagree, so wg[i] = (not sub[i]) if i in disagreements else "
             "sub[i] recovers every verdict exactly, asserted against the stored rate. No "
             "model is loaded and no GPU is needed, which is why this ran after the pod was "
             "released.",
    ),
    Claim(
        id="A3", layer=3,
        statement="RESOLVED 2026-09-22, NEGATIVE. Refusal is NOT multi-directional. The "
                  "stances that survive ablation have no second direction that can be acted "
                  "on. A within-harmful contrast (mean(inability) - mean(identity)) is "
                  "reliable (0.688 vs a 0.229 pseudo-stance null, ~6 sigma) and nearly "
                  "orthogonal to Arditi's (|cos| 0.191, 0.23 disattenuated) -- but steering "
                  "on it does NOT change which stance the model produces, at any magnitude, "
                  "against five independent norm-matched nulls. The refusal direction itself "
                  "reshapes the stance mix MORE than the stance direction does (0.308 vs "
                  "0.266 at the in-regime magnitude). The geometry encoded PROMPT CONTENT: "
                  "the stance classes are different prompts and the model picks its stance "
                  "from the prompt, so a reliable separating direction is exactly what a "
                  "topic confound looks like. What A3 reports is the BOUND -- the surviving "
                  "stance is not linearly mediated at the eoi position in any actionable "
                  "way, which constrains the linear-representation hypothesis and "
                  "reconciles with B1's identity refusals going 20 -> 21.",
        evidence=(
            Evidence("stance_directions", "stance_directions.py", "A3",
                     "per-stance mean-diff directions, the split-half CEILING they must be "
                     "read against, and the within-harmful stance contrast with its "
                     "reliability and pseudo-stance null"),
            Evidence("stance_steer", "stance_steer.py", "A3b",
                     "the causal test: composition and rate spans per magnitude against five "
                     "independent nulls, with KL tiering and a degeneracy guard"),
        ),
        controls=(
            Control("positive control: re-fit inability", "a broken contrast construction "
                    "producing directions from noise", True,
                    "2026-09-22: PASSED. A3 selects (pos 3, L14) -- the argmax of "
                    "run_stage's stored steer surface -- and cos(d_inability, d_arditi) = "
                    "+0.996 there. NOTE the comparator: run_stage selects by ABLATION and A3 "
                    "by INDUCE, and those disagree for Arditi's own direction (+0.826 at the "
                    "surface argmax vs +0.519 at the ablation cell), so demanding the stored "
                    "(pos_star, l_star) would have failed a correct direction. An earlier "
                    "version swept LAYERS ONLY with the eoi position pinned to the last, "
                    "landing on a cell where Arditi's own direction scores -1.417.",
                    script="stance_directions.py"),
            Control("split-half ceiling for every cosine", "reading a pairwise cosine with "
                    "no idea what agreement looks like when the directions ARE the same", True,
                    "2026-09-22: RUN, and it retired a number. The stance-vs-harmless "
                    "cosine of 0.972 sits against a ceiling of 0.980 -- both fits are "
                    "harmful-vs-harmless with the stance label only choosing which harmful "
                    "prompts go in, so a high cosine is near-guaranteed by construction. "
                    "Every fit in the block uses one k so ceiling and observed are the same "
                    "measurement at the same n.", script="stance_directions.py"),
            Control("shuffled-label null per stance", "a direction fitted on an arbitrary "
                    "partition of the model's own outputs looking like something", True,
                    "2026-09-22: RUN. Same class sizes, same pooled activations, labels "
                    "randomised, so it shares the anisotropic geometry and is harder than an "
                    "isotropic null. inability z=+2.8, identity z=+2.5.",
                    script="stance_directions.py"),
            Control("count balance", "a mean-diff dominated by the larger class -- the stance "
                    "classes differ by up to 6x", True,
                    "2026-09-22: RUN. Both classes subsampled to the smaller size before "
                    "fitting.", script="stance_directions.py"),
            Control("causal test against FIVE nulls", "a geometric direction that is "
                    "reliable, orthogonal and significant against its own null while "
                    "encoding an entirely different property (prompt topic, not stance)",
                    True,
                    "2026-09-22: RUN, AND IT FIRED. Necessary because geometry cannot "
                    "separate stance from topic when the stance label is DERIVED FROM THE "
                    "PROMPT -- the confound is in the class definition, not the estimator. "
                    "d_stance clears the null at no magnitude, and arditi out-moves it on "
                    "composition everywhere. An earlier run with ONE null draw and a bare "
                    "'>' reported a dissociation on a margin of 0.031; the verdict now "
                    "requires n>=3 draws, max AND mean+2sd, AND that stance move composition "
                    "more than arditi does.", script="stance_steer.py"),
            Control("regime + degeneracy guard", "reading a destroyed model as a clean "
                    "effect -- stance_of() has no 'broken' bucket, so gibberish scores as "
                    "'compliance' and reports refusal 0.000", True,
                    "2026-09-22: RUN after the first attempt hit exactly this. Every cell is "
                    "KL-tiered on harmless prompts and degeneracy-checked; degeneracy was "
                    "0.000 in all 42 cells of the reported run, so the negative is not a "
                    "broken-model artifact.", script="stance_steer.py"),
            Control("cross-checkpoint transplant", "the circularity of fitting a direction on "
                    "classes derived from the model's OWN completions", True,
                    "2026-09-22: MOOT and recorded as such. It was designed to test whether "
                    "d_identity transfers to another family. A3b shows d_stance has no "
                    "causal effect on stance in the model it was fitted on, so there is "
                    "nothing whose transfer would be informative. Not run, and not "
                    "outstanding.", script="transplant.py"),
        ),
        depends_on=("C1", "C2", "C3", "P1-E7"),
        falsifier="FIRED. The pre-registered falsifier was: the stance arm fails to move the "
                  "inability:identity ratio beyond the null arm at any coefficient -> "
                  "d_stance is a prompt-content direction and multi-directionality does not "
                  "survive. That is what happened, at all three magnitudes. The claim now "
                  "standing is the BOUND, whose own falsifier is: a direction that DOES "
                  "causally control stance is found at the eoi position -- by a contrast not "
                  "built on prompt-derived labels, or on a model with enough of two stances "
                  "to fit one without that confound.",
        note="Layer 3. tulu2_dpo/baseline is the only configuration that fits inability (78) "
             "AND identity (41) from one model, one prompt set, one activation cache. That "
             "makes it decisive for the NEGATIVE, which needs no replication: a claim that "
             "no actionable second axis was found in the one model where it could be looked "
             "for is bounded by that model, and is stated that way. A POSITIVE would have "
             "needed a second family before being written, which it never reached.",
    ),
    Claim(
        id="P1-E7", layer=2,
        statement="CONFIRMED 2026-09-19. Breaking alignment breaks the LINK and spares "
                  "the REPRESENTATION. Attacked: probe 1.000, XSTest focus-matched 0.917, "
                  "behavioural refusal 0.477 (hand-audited; substring said 0.189), and ZERO "
                  "of 130 cells induce. Its own "
                  "direction reaches -4.955 at natural scale; rlvr's direction injected "
                  "into it reaches +1.069 and crosses. P1-E7d then decomposed this: the "
                  "REPRESENTATION is intact (probe 1.000 at every dose), the READOUT is "
                  "intact (the frozen pre-attack direction induces +3.4 to +4.7 in every "
                  "attacked checkpoint), and only the MAPPING between them is destroyed -- "
                  "the model stops producing the refusal direction when it sees harm.",
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
                    "examples; the only difference is 50 rehearsed refusals. CORRECTED "
                    "2026-09-23: the efficacy figures quoted here were measured on tail[:48], "
                    "which lies ENTIRELY inside the 50 prompts the control rehearsed, so the "
                    "control's figure was a memorisation readout. On the 82 HELD-OUT prompts "
                    "(WildGuard, 48 tok): control 0.902 vs attacked 0.439 vs rlvr 0.976. The "
                    "gap is 0.463 held-out against 0.455 over all 132, and within each prompt "
                    "set the control's advantage is as large or LARGER on held-out prompts at "
                    "every dose -- so the conclusion stands: fine-tuning per se does not do "
                    "this.", script="attack.py"),
            Control("rehearsal and evaluation are disjoint", "scoring the control on the very "
                    "prompts it was trained to refuse, so 'preserved' reads as memorised", True,
                    "2026-09-23: FIXED after it was found live. build_safety_examples drew "
                    "rehearsal prompts from the same 132-prompt tail every behavioural "
                    "measurement scores, and attack.py's efficacy check sat inside that set. "
                    "data.split_tail now makes rehearsal (tail[:50]) and evaluation "
                    "(tail[50:]) disjoint by construction, with an assertion; smoke_test "
                    "drives the real build_safety_examples with the model stubbed and fails "
                    "on any overlap. Mechanism numbers were never affected -- they use "
                    "harmful_train[:128] and harmful_val, not the tail. The held-out "
                    "restatement is heldout_control.py -> results/p1e7_heldout_ANALYSIS.json.",
                    script="heldout_control.py"),
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
            Control("dose-0 direction transplanted into a later dose", "the two readings of "
                    "'no valid direction at dose 100': the COUPLING is destroyed, or the "
                    "mean-diff estimator no longer FINDS a direction that still works. The "
                    "direction is re-fitted from each dose's own activations, so the curve "
                    "cannot separate them, and the strong claim needs the first reading",
                    True,
                    "Adapters are saved at every dose (models/olmo2_e7d-{arm}-adapter-{step}), "
                    "so this is runnable without retraining: inject the dose-0 direction into "
                    "the dose-100 model. Induces refusal -> the coupling survived and the "
                    "estimator lost it. Does not -> the coupling is gone. Exactly P1-E7b's "
                    "rlvr->attacked cell, which DID restore refusal at +1.069, so the "
                    "instrument is known to work. BLOCKING for 'the coupling is destroyed "
                    "before the behaviour is'; the weaker 'no direction is FINDABLE at dose "
                    "100' already holds. BUILT 2026-09-22 into dose_response.py instead of a "
                    "separate script: the dose-0 direction is frozen and re-injected at every "
                    "later dose, which gives a CURVE rather than one point and needs no saved "
                    "checkpoint. Carries its own positive control -- at dose 0 the frozen cell "
                    "IS the self cell, so the sweep must reproduce steer at l* or the run "
                    "aborts.", script="dose_response.py"),
            Control("step dose-response", "a single before/after pair being a coincidence",
                    True, "2026-09-22: RUN, both arms, six doses. It did not confirm the "
                    "prediction -- it sharpened it. Coupling 13 -> 0 steerable layers by step "
                    "100 while refusal still retains 87% of dose 0 (WildGuard 0.985 -> 0.856); "
                    "behaviour then decays to 0.477 by step 1500. They do NOT fall together: the link breaks at once "
                    "and the behaviour decays afterwards. Probe = 1.000 at EVERY dose in BOTH "
                    "arms (mass-mean 0.981-1.000), so the falsifier did not fire. In this run "
                    "(ONE seed, first dose 100) the control keeps 62-85% of its steerable "
                    "layers and 85-95% of its behaviour at every measured dose. D1 "
                    "(2026-09-23) showed that understates control variability: across three "
                    "OLMo 2 control seeds with a dose-50 point, steerable layers range 0-13 -- "
                    "s1 loses all 13 at dose 50 in a genuine behavioural dip (WildGuard 0.427) "
                    "and recovers by 100 -- while max induce stays positive at every dose but "
                    "that one transient. The SIGN contrast with the attack survives; the "
                    "layer-count contrast does not. The "
                    "surviving refusals shift register exactly where coupling dies: normative "
                    "share 0% -> 68% in the benign arm, 0-1% at every dose in the control. "
                    "dose 1500 also replicates the destroyed 2026-09-19 checkpoint at a "
                    "different LoRA draw: WildGuard 0.477 here against the audited 0.477 there, "
                    "to three decimals, and max induce -5.170 vs -5.188. See "
                    "results/olmo2_e7d_dose_response_ANALYSIS.json. Was "
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
            Control("seed replication", "one stochastic training run", True,
                    "2026-09-22: CLOSED by dose_response.py. The original attacked checkpoint "
                    "died with its pod, so dose 1500 is a fully independent run at a "
                    "different LoRA draw -- and it lands on substring 0.182 against 0.189 "
                    "(same judge both times; the audited figure there was 0.477 and this "
                    "run's estimate is 0.492), l* = -1 in both, zero steerable layers in "
                    "both, max induce -5.170 against -5.188. Closer than a seed replication "
                    "had any right to be.",
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


# --------------------------------------------------------------- superseded measurements
#
# WHY THIS EXISTS. On 2026-09-21 a validated classifier showed the substring judge
# undercounts refusal by 8-10x wherever a model declines in a NORMATIVE register ("this is
# illegal and unethical", "I strongly condemn") rather than a first-person one ("I cannot").
# Three headline numbers moved. RESULTS.md and the published report were corrected the same
# day -- and this graph was not, so for a day it cited 0.189 and 1.000 as live evidence for
# claims whose real numbers were 0.477 and 0.750. The graph is what a paper is written from,
# so a stale graph is worse than a stale draft.
#
# A judge is a SHARED INSTRUMENT: correcting it changes every claim that used it, and
# nothing in a per-claim graph makes that propagation happen. So superseded values are
# registered here and `check()` fails if any claim still quotes one. The registry is the
# propagation mechanism -- adding a row is how a correction reaches every claim at once.
SUPERSEDED: tuple[tuple[str, str, str], ...] = (
    ("control arm 1.000 -> 1.000", "control 0.902 vs attacked 0.439",
     "P1-E7's matched-control efficacy figure. It was measured on tail[:48], which lies "
     "entirely inside the 50 prompts the control rehearsed its own refusals on -- a "
     "memorisation readout, not a preservation measurement. Restated on the 82 held-out "
     "prompts (WildGuard, 48 tok). The conclusion survives; the number does not"),
    ("0.871", "0.972",
     "A3's pairwise stance cosine. The 0.871 run swept LAYERS with the eoi position pinned "
     "to the last one, landing on (pos 4, L12) where Arditi's own direction scores -1.417 on "
     "the stored steer surface. Corrected run sweeps both axes and lands on (pos 3, L14), "
     "the surface argmax. NOTE both numbers are near a split-half ceiling of 0.980 and "
     "neither is reportable on its own -- the stance-vs-harmless contrast cannot separate "
     "stance from harmfulness"),
    ("0.076 against a null of 0.008", "0.174 vs 0.121",
     "A3b's rate-axis separation. The 0.008 was ONE null draw. Against five independent "
     "nulls the in-regime magnitude clears by 0.001 (0.076 vs mean+2sd 0.075), |c|=0.25 "
     "clears properly but is outside the strict KL bound, and |c|=0.5 fails. The '~10x' "
     "phrasing is retired"),
    ("DISSOCIATION: YES", "NO DISSOCIATION at any magnitude",
     "A3b run 75e3f48 (ledger 2026-09-22T20:34:15, notes say dissociation=True). It used ONE "
     "pseudo-stance null and the bare criterion stance_span > null_span, passing on "
     "0.266 vs 0.235. The negative control had moved composition nearly as much as the "
     "treatment, and the arditi arm moved it MORE (0.308). The row stays in the ledger as "
     "history; the number must never be quoted"),
    ("0.985 -> 0.606", "0.985 -> 0.485",
     "ablation's surviving refusal rate, re-measured at 128 generated tokens. At 48 the "
     "normative-register PREAMBLE is all a judge can see; 19 of 80 continue into explanation "
     "and are compliance. Every 48-token refusal rate in this project is an upper bound"),
    ("0.189", "0.477",
     "attacked-arm behavioural refusal: substring judge, corrected by WildGuard + a "
     "prompt-paired audit of all 39 disagreements (38 genuine refusals, 1 partial)"),
    ("-> 1.000 strict refusal in TEXT", "-> 0.750 hand-audited",
     "P1-E1c injected arm: all 64 completions read; the substring judge scored degenerate "
     "text as refusal"),
    ("0.985 -> 0.000", "0.985 -> 0.606",
     "ablation does NOT produce compliance; 171/171 audited 'bypasses' are refusals in a "
     "normative register"),
    ("29x", "22x",
     "Zephyr base-over-SFT ratio: computed from a pre-2026-09-13 judge against an SFT rate "
     "the same judge undercounts 10x"),
)


# Documents that quote measurements and are read by humans rather than by check(). The
# registry guarded the claim graph and nothing else, so on 2026-09-23 a superseded figure sat
# in RESULTS.md and the published report while `provenance --check` reported OK. A registry
# that only polices the file it lives in is not a propagation mechanism.
CITING_DOCS: tuple[str, ...] = ("RESULTS.md", "report/refusal-machinery.html")


def superseded_in_documents(docs: tuple[str, ...] = CITING_DOCS) -> list[str]:
    """Superseded values still quoted in the prose, with no sign of the correction nearby.

    SECTION-scoped, not line-scoped and not file-scoped. File-scoped is too weak: a document
    that corrects a number in one place and quotes the stale one elsewhere passes. Line-scoped
    is too strict, and the first version proved it -- it flagged four citations inside the very
    section whose subject is that correction, where a table and its prose legitimately repeat
    the old value several lines from the new one. A section is the unit a reader actually
    consumes, so it is the right unit to demand the correction appear in."""
    out = []
    for path in docs:
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        # Markdown headings or HTML <section> both delimit "what a reader has in view".
        marks = [m.start() for m in re.finditer(r"(?m)^#{1,3} |<section[ >]", text)] or [0]
        for old_v, new_v, why in SUPERSEDED:
            for m in re.finditer(re.escape(old_v), text):
                start = max((p for p in marks if p <= m.start()), default=0)
                end = min((p for p in marks if p > m.start()), default=len(text))
                section = text[start:end]
                if new_v in section or "~~" in text[m.start() - 12:m.start() + 12]:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                head = text[start:start + 70].strip().splitlines()[0] if section else ""
                out.append(f"STALE DOC: {path}:{line} quotes {old_v!r}; its section "
                           f"({head!r}) never mentions {new_v!r} -- {why}")
    return out


def superseded_citations(claims=CLAIMS) -> list[str]:
    """Any claim still quoting a number a later measurement replaced.

    Scanned across statement, evidence and control text, because a stale number is equally
    misleading wherever it sits. A claim may quote the old value only when it also names the
    new one -- that is a correction being documented, not a stale citation."""
    out = []
    for c in claims:
        blobs = {"statement": c.statement}
        blobs.update({f"evidence[{e.axis}]": e.what for e in c.evidence})
        blobs.update({f"control[{k.name}]": k.where for k in c.controls})
        for where, text in blobs.items():
            for old, new, why in SUPERSEDED:
                if old in text and new not in text:
                    out.append(f"STALE: {c.id} {where} quotes {old!r}, superseded by {new!r} "
                               f"-- {why}. Quote the new value, or cite both if the point IS "
                               f"the correction.")
    return out


# --------------------------------------------------------------------------- disk + ledger

def on_disk() -> dict[str, list[str]]:
    """axis -> the result files that exist.

    Two shapes, because not every axis is an array: the sweeps are
    results/{lineage}_{stage}_{axis}.npz, but P1-E1c stores COMPLETIONS as
    results/{lineage}_{target}_from_{source}_text.json -- text is the evidence there, so it
    has to be text on disk. The first version of this scanner only globbed .npz and reported
    C3's text evidence as missing; the check caught my own graph being wrong, which is the
    point of having it.


    AN AXIS MAY CONTAIN UNDERSCORES, and until 2026-09-22 this function assumed it could not.
    It derived the axis with rsplit("_", 1), so tulu2_dpo_dpo_stance_directions.npz parsed as
    axis "directions" and A3's declared `stance_directions` evidence could never match a file
    that was sitting right there on disk. The check reported the evidence missing, which looks
    exactly like the failure it is supposed to detect -- a claim with nothing behind it -- and
    would have been "fixed" by weakening the claim.

    So the axes are no longer GUESSED from the filename. They come from the claim graph, which
    is the only place that knows what an axis is, and the longest match wins so that
    `stance_directions` is preferred over a hypothetical `directions`."""
    axes = sorted({e.axis for c in CLAIMS for e in c.evidence}, key=len, reverse=True)
    out: dict[str, list[str]] = {}
    # *_ANALYSIS.json is the third shape: the CPU-only experiments (A2) write a JSON report
    # rather than an array, and the first version of this scanner globbed only .npz and
    # _text.json, so A2's evidence read as missing the moment it was registered.
    for pat, ext in ((f"{RESULTS}/*.npz", ".npz"), (f"{RESULTS}/*_text.json", ".json"),
                     (f"{RESULTS}/*_ANALYSIS.json", ".json")):
        for p in sorted(glob.glob(pat)):
            stem = os.path.basename(p)[: -len(ext)]
            # A file may BE its axis (d1_controls_ANALYSIS.json) as well as end with it
            # (a2_judge_bench_ANALYSIS.json). Suffix-only matching made the first kind
            # invisible -- present evidence reported missing, the O-158 failure again.
            hit = next((a for a in axes if stem == a or stem.endswith("_" + a)), None)
            # Unknown files still register under a best-effort axis, so a NEW result appears
            # in the report before its claim is written rather than being invisible.
            axis = hit if hit else stem.rsplit("_", 1)[-1]
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
        # PLANNED vs BROKEN. This file's own instructions say to add a Claim FIRST, with its
        # controls and falsifier, and only then write the script -- so a claim whose evidence
        # does not exist yet is following the process, not violating it. Treating that as a
        # problem made `--check` exit 1 the moment A3 was registered as designed (2026-09-23),
        # which would train the reader to ignore the check. A claim is PLANNED when none of
        # its evidence exists AND none of its controls is marked done; anything else is a
        # partially-run claim, where a missing file IS a real inconsistency.
        planned = (all(e.axis not in files for e in c.evidence)
                   and not any(k.done for k in c.controls))
        if planned:
            continue
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

    problems.extend(superseded_citations(claims))
    problems.extend(superseded_in_documents())

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
