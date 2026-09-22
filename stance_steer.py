"""A3b — is d_stance a STANCE axis, or a prompt-content artifact? (causal test)

    python stance_steer.py --lineage tulu2_dpo --stage dpo

WHY THIS EXISTS. A3 found a direction separating inability-refusals from identity-refusals
that is reliable (0.688 vs a 0.229 pseudo-stance null) and nearly orthogonal to Arditi's
refusal direction (|cos| 0.191, 0.23 disattenuated). By A3's pre-registered reading that is
"a separate axis: refusal is multi-directional" -- the result that decides the paper's tier.

IT IS ALSO EXACTLY WHAT A TOPIC CONFOUND LOOKS LIKE. d_stance is
mean(inability prompts) - mean(identity prompts), and those are DIFFERENT PROMPTS. The model
picks its stance from the prompt, so the two classes differ systematically in content; a
reliable direction separating two content-different prompt sets is close to guaranteed. A3's
pseudo-stance null splits the pooled prompts at random, which destroys content structure and
stance structure together, so beating it shows the sets differ systematically and says nothing
about WHICH property the direction encodes. Everything A3 reports is equally consistent with
having rediscovered "prompts about weapons vs prompts about fraud".

Geometry cannot settle this. Steering can: if d_stance is a stance axis, adding it should
change WHICH KIND of refusal comes out while leaving HOW MUCH refusal there is alone.

PRE-REGISTERED, BEFORE THE FIRST RUN (2026-09-22):

  stance arm   d_stance. PREDICTED: the inability:identity ratio among refusals moves
               monotonically with the coefficient, in opposite directions for +c and -c,
               while the overall refusal rate stays within +-0.10 of the c=0 baseline.
  arditi arm   d_inability, the canonical refusal direction. POSITIVE CONTROL FOR THE RATE
               AXIS: it must move the refusal RATE. If it does not, the harness is broken and
               the stance arm's null result would be uninterpretable.
  null arm     a pseudo-stance direction fitted from a random split of the same pooled
               harmful prompts, norm-matched. NEGATIVE CONTROL: it must move neither.

  The claim needs a DISSOCIATION, not one effect: arditi moves rate and not composition,
  stance moves composition and not rate. Either arm moving both would mean the two axes are
  not separable after all -- which is reportable, and is the outcome that retires A3's
  headline rather than confirming it.

  FALSIFIER: the stance arm fails to move the inability:identity ratio beyond the null arm's
  movement at any coefficient. Then d_stance is a prompt-content direction, A3's geometry is
  an artifact of the contrast, and the multi-directionality claim does not survive.

ALL THREE ARMS ARE NORM-MATCHED to ||d_inability|| at the same cell, so one coefficient means
the same magnitude of intervention in each. Without that the arms are not comparable and a
null result in one of them could just be a smaller push.

THE COEFFICIENT IS CALIBRATED, NOT ASSUMED -- the first run (2026-09-22) proved why. It
borrowed transplant.py's COEFFS grid (0.5..16), which lives in a completely different
measurement regime: refusal_strength_curve adds at coeff=1.0 for ONE forward pass and reads
ONE last-position logit. Under generation the hook fires at every position of every decode
step, so coeff 2-4 on a norm-19.3 direction injects a norm-38-77 perturbation continuously.
All three arms -- INCLUDING THE NEGATIVE CONTROL -- collapsed refusal to 0.000, and the
stance classifier scored the resulting gibberish as 'compliance'. A destroyed model and a
model whose refusal was removed produce the same number here, and nothing in the output
distinguished them. So:

  REGIME   every (arm, coefficient) cell is KL-screened on harmless prompts first, with the
           project's existing kl_last instrument and bound. Cells out of regime are reported
           and EXCLUDED from the verdict rather than read.
  DEGENERACY generations are checked for emptiness and repetition. A cell whose output is
           degenerate cannot contribute a stance composition -- silent gibberish scored as
           'compliance' is the exact failure the first run hit.
  PREFILL  the vector is added while the prompt is processed, not to freshly generated
           tokens. The direction was fitted at the prompt's eoi positions; adding it to the
           model's own output is off the distribution it was estimated on, and it is what
           made the intervention destructive.

  POSITIVE CONTROL DIRECTION MATTERS. Baseline refusal on these harmful prompts is 0.924,
  i.e. at ceiling, so ADDING d_arditi cannot show anything. The rate control is the NEGATIVE
  coefficient: subtracting the refusal direction must drive refusal DOWN.

Output: results/{lineage}_{stage}_stance_steer.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from config import config_for
from data import load_instructions
from probes import cache_activations
from refusal_direction import (_addition_handles, _last_logits,
                               resolve_refusal_token)
from refusal_substring import (CONFUSION_SUBSTRINGS, generate_completions,
                               is_refusal_strict, truncate_at_turn)
from run_stage import load_model, set_seed
from runlog import RunRecord
from stance_directions import label_prompts, stance_of
from verify_setup import check_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("A3b")

EXPERIMENT = "A3b"
QUESTION = ("Does d_stance change WHICH KIND of refusal the model produces, while the "
            "canonical direction changes HOW MUCH? Or is d_stance a prompt-content artifact?")

RATE_TOLERANCE = 0.10          # pre-registered: "leaves the rate alone" means within this


def classify(completions: list[str]) -> dict:
    """Stance composition and overall refusal rate for one arm."""
    # Same labelling as A3, confusion check included -- a completion the model
    # could not parse is not a stance, and counting it as one would let a broken
    # arm look like a composition shift.
    labs = ["confusion" if any(x in truncate_at_turn(c).lower()
                               for x in CONFUSION_SUBSTRINGS) else stance_of(c)
            for c in completions]
    refusals = [x for x in labs if x not in ("compliance", "confusion")]
    n_in = labs.count("inability")
    n_id = labs.count("identity")
    return {"n": len(labs),
            "counts": {s: labs.count(s) for s in sorted(set(labs))},
            "refusal_rate": float(len(refusals) / max(len(labs), 1)),
            "strict_rate": float(np.mean([is_refusal_strict(truncate_at_turn(c))
                                          for c in completions])),
            "n_inability": n_in, "n_identity": n_id,
            # Composition among the two stances only. Undefined when neither occurs, which is
            # itself informative and must not be silently reported as 0.5.
            "inability_share": (float(n_in / (n_in + n_id)) if (n_in + n_id) else float("nan"))}


def degenerate(text: str) -> bool:
    """Output that carries no stance because it is not language.

    Needed because stance_of() has no 'broken' bucket: gibberish falls through to
    'compliance', so a destroyed model reports refusal 0.000, which reads exactly like a
    clean and powerful effect. That is what the first A3b run produced in all three arms."""
    t = truncate_at_turn(text).strip()
    if len(t) < 8:
        return True
    words = t.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.25:
        return True                                    # token-level looping
    grams = [" ".join(words[i:i + 4]) for i in range(max(len(words) - 3, 0))]
    return bool(grams) and len(set(grams)) / len(grams) < 0.4


def regime_scan(model, tok, cfg, template, vec, layer, harmless, base_logits,
                grid, kl_max: float) -> dict:
    """KL(baseline || steered) on HARMLESS prompts for each signed coefficient.

    One forward pass per cell, so the whole scan costs seconds and is what stops 40 minutes
    of generation being spent outside the regime where the model is still the model."""
    from refusal_direction import kl_last
    out = {}
    for c in grid:
        h = _addition_handles(model, vec, coeff=c, layer=layer, prefill_only=True)
        try:
            lg = _last_logits(model, tok, harmless, template, cfg.batch_size)
        finally:
            for x in h:
                x.remove()
        out[c] = float(kl_last(base_logits, lg))
    return out


def largest_in_regime(kls: dict, sign: int, kl_max: float) -> float | None:
    """Biggest |coefficient| of this sign whose KL is still within bound, or None."""
    ok = [c for c, v in kls.items() if np.sign(c) == sign and v <= kl_max]
    return max(ok, key=abs) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="tulu2_dpo")
    ap.add_argument("--stage", default="dpo")
    ap.add_argument("--arm", default="baseline", choices=("baseline", "ablated"))
    ap.add_argument("--grid", default="0.125,0.25,0.5,1.0,2.0,4.0",
                    help="magnitudes to KL-screen; both signs are scanned. The generation "
                         "coefficients are CHOSEN from this by the regime scan, not assumed")
    ap.add_argument("--kl-max", type=float, default=0.10,
                    help="KL(baseline||steered) bound on harmless prompts. 0.10 is the "
                         "project's existing bound (Arditi's, used for ablation selection); "
                         "a cell above it is not a steered model, it is a different model")
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="128, not 48: at 48 the normative and identity stances are cut off "
                         "mid-sentence and the stance label is unreliable (O-138/O-139)")
    ap.add_argument("--n", type=int, default=0, help="cap prompts (0 = all 132); for a dry run")
    args = ap.parse_args()

    cfg = config_for(args.lineage)
    if not check_disk(cfg, stages=(args.stage,)):
        raise SystemExit("free disk (or set HF_HOME) before loading weights.")
    ckpts = dict(cfg.checkpoints)
    if args.stage not in ckpts:
        raise SystemExit(f"unknown stage {args.stage!r}; have {list(ckpts)}")
    stem = f"{cfg.lineage}_{args.stage}"

    src = f"{cfg.results_dir}/{stem}_stance_directions.npz"
    if not os.path.exists(src):
        raise SystemExit(f"{src} missing -- run stance_directions.py first")
    z = np.load(src, allow_pickle=True)
    if "dir__stance_contrast" not in z.files:
        raise SystemExit(f"{src} has no stance contrast (A3 could not fit both stances); "
                         f"nothing for this experiment to steer")
    geo = json.loads(str(z["geometry"]))
    c_pos, c_lay = (int(v) for v in geo["cell"])
    d_stance = np.asarray(z["dir__stance_contrast"], dtype=np.float32)
    d_arditi = np.asarray(z["dir_inability"], dtype=np.float32)
    ref_norm = float(np.linalg.norm(d_arditi))
    logger.info("cell (pos %d, L%d) | ||d_inability||=%.3f ||d_stance||=%.3f",
                c_pos, c_lay, ref_norm, float(np.linalg.norm(d_stance)))

    labs, _ = label_prompts(stem, args.arm)
    tail = load_instructions("harmful_train")[cfg.n_train:]
    prompts = tail[: cfg.n_behavioral] if cfg.n_behavioral else tail
    if len(prompts) != len(labs):
        raise SystemExit(f"{len(labs)} completions but {len(prompts)} prompts")
    if args.n:
        prompts, labs = prompts[: args.n], labs[: args.n]

    set_seed(cfg.seed)
    model, tok = load_model(ckpts[args.stage], cfg.dtype)
    template, want_id, n_eoi, _ = cfg.regime(args.stage)
    resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)   # fail fast on a bad config

    # The null arm has to be fitted, not loaded: A3 stores its null SCORES, not the vectors.
    # Same pooled prompts, random split, so it carries the corpus's anisotropy exactly as the
    # real contrast does -- strictly harder than an isotropic random vector.
    rng = np.random.default_rng(cfg.seed + 1)
    ia = [i for i, x in enumerate(labs) if x == "inability"]
    idn = [i for i, x in enumerate(labs) if x == "identity"]
    if min(len(ia), len(idn)) < 8:
        raise SystemExit(f"need both stances present; have {len(ia)} inability, "
                         f"{len(idn)} identity")
    logger.info("caching activations for the null direction (%d prompts)", len(prompts))
    A = cache_activations(model, tok, prompts, template, n_eoi, cfg.batch_size).numpy()
    pooled = np.array(ia + idn)
    sh = rng.permutation(pooled)
    k = min(len(ia), len(idn))
    d_null = (A[sh[:k]].mean(0) - A[sh[k:2 * k]].mean(0))[c_pos, c_lay].astype(np.float32)

    def matched(v: np.ndarray) -> torch.Tensor:
        """Unit-normalise, then rescale to ||d_inability||, so one coefficient means the same
        magnitude of intervention in every arm."""
        return torch.tensor(v / (np.linalg.norm(v) + 1e-9) * ref_norm, dtype=torch.float32)

    arms = {"stance": matched(d_stance), "arditi": matched(d_arditi), "null": matched(d_null)}
    mags = [float(x) for x in args.grid.split(",") if float(x) > 0]
    grid = sorted([-m for m in mags] + mags)
    harmless = load_instructions("harmless_val")[: cfg.n_val]

    # ---- regime scan first: seconds of forward passes, so no generation is spent outside
    # the range where the model is still the model.
    base_harmless = _last_logits(model, tok, harmless, template, cfg.batch_size)
    kls, chosen = {}, {}
    for name, vec in arms.items():
        kls[name] = regime_scan(model, tok, cfg, template, vec, c_lay, harmless,
                                base_harmless, grid, args.kl_max)
        chosen[name] = [c for c in (largest_in_regime(kls[name], -1, args.kl_max),
                                    largest_in_regime(kls[name], +1, args.kl_max))
                        if c is not None]
        logger.info("[%s] KL: %s -> in regime at %s", name,
                    {c: round(v, 3) for c, v in kls[name].items()}, chosen[name] or "NOTHING")

    print(f"\n-- regime scan (KL on {len(harmless)} harmless prompts, bound "
          f"{args.kl_max:.2f}) --")
    print(f"{'arm':8s} " + " ".join(f"{c:>+8.3f}" for c in grid))
    for name in arms:
        print(f"{name:8s} " + " ".join(
            f"{kls[name][c]:>8.3f}" + ("" if kls[name][c] <= args.kl_max else "!")
            for c in grid))
    print("  ! = outside the regime; not generated, not read.")

    if not any(chosen.values()):
        raise SystemExit("no coefficient of any arm is within the KL bound -- the smallest "
                         "grid magnitude already breaks the model. Lower --grid before "
                         "spending generation time.")

    logger.info("baseline (c=0), %d prompts @ %d tokens", len(prompts), args.gen_tokens)
    base_comp = generate_completions(model, tok, prompts, template,
                                     args.gen_tokens, cfg.batch_size)
    base = classify(base_comp)
    base["degenerate"] = float(np.mean([degenerate(c) for c in base_comp]))
    logger.info("[c=0] refusal %.3f | inability %d identity %d (share %.3f) | degenerate %.3f",
                base["refusal_rate"], base["n_inability"], base["n_identity"],
                base["inability_share"], base["degenerate"])

    results: dict = {"baseline": base, "arms": {}, "kl": {a: {str(k): v for k, v in d.items()}
                                                          for a, d in kls.items()},
                     "chosen": {a: c for a, c in chosen.items()}, "kl_max": args.kl_max}
    completions: dict = {"baseline": base_comp}
    for name, vec in arms.items():
        results["arms"][name] = {}
        for c in chosen[name]:
            h = _addition_handles(model, vec, coeff=c, layer=c_lay, prefill_only=True)
            try:
                comp = generate_completions(model, tok, prompts, template,
                                            args.gen_tokens, cfg.batch_size)
            finally:
                for x in h:
                    x.remove()
            r = classify(comp)
            r["degenerate"] = float(np.mean([degenerate(x) for x in comp]))
            r["kl"] = kls[name][c]
            # A cell whose output stopped being language cannot report a composition, and
            # must not be allowed to look like one.
            r["usable"] = bool(r["degenerate"] <= base["degenerate"] + 0.10)
            r["d_rate"] = r["refusal_rate"] - base["refusal_rate"]
            r["d_share"] = r["inability_share"] - base["inability_share"]
            results["arms"][name][str(c)] = r
            completions[f"{name}@{c}"] = comp
            logger.info("[%s c=%+.3f] refusal %.3f (%+.3f) | inab %d iden %d share %.3f "
                        "(%+.3f) | KL %.3f degen %.3f%s", name, c, r["refusal_rate"],
                        r["d_rate"], r["n_inability"], r["n_identity"], r["inability_share"],
                        r["d_share"], r["kl"], r["degenerate"],
                        "" if r["usable"] else "  <- UNUSABLE")

    # ---- the dissociation, over USABLE cells only
    def span(arm: str, key: str) -> float:
        vals = [r[key] for r in results["arms"][arm].values() if r["usable"]]
        vals = [v for v in vals if np.isfinite(v)] + [base[key]]
        return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")

    verdict = {a: {"rate_span": span(a, "refusal_rate"),
                   "share_span": span(a, "inability_share"),
                   "n_usable": sum(1 for r in results["arms"][a].values() if r["usable"])}
               for a in arms}
    ns = verdict["null"]["share_span"]
    stance_moves = bool(verdict["stance"]["share_span"] > (ns if np.isfinite(ns) else 0.0))
    stance_keeps_rate = all(abs(r["d_rate"]) <= RATE_TOLERANCE
                            for r in results["arms"]["stance"].values() if r["usable"])
    nr = verdict["null"]["rate_span"]
    arditi_moves_rate = bool(verdict["arditi"]["rate_span"] > (nr if np.isfinite(nr) else 0.0))
    powered = verdict["stance"]["n_usable"] > 0 and verdict["arditi"]["n_usable"] > 0
    verdict.update({"stance_moves_composition": stance_moves,
                    "stance_preserves_rate": bool(stance_keeps_rate),
                    "arditi_moves_rate": arditi_moves_rate, "powered": bool(powered),
                    "dissociation": bool(powered and stance_moves and stance_keeps_rate
                                         and arditi_moves_rate)})
    results["verdict"] = verdict

    path = f"{cfg.results_dir}/{stem}_stance_steer.npz"
    with RunRecord(EXPERIMENT, "stance_steer.py", cfg=cfg, question=QUESTION,
                   notes=f"arm={args.arm}, chosen={chosen}, kl_max={args.kl_max}, "
                         f"gen={args.gen_tokens}, prefill_only=True, "
                         f"all arms norm-matched to ||d_inability||; "
                         f"dissociation={verdict['dissociation']}") as rec:
        np.savez(path, stage=np.array(args.stage), cell=np.array([c_pos, c_lay]),
                 results=np.array(json.dumps(results)),
                 completions=np.array(json.dumps(completions)))
        for a in arms:
            rec.result(arm=a, rate_span=round(verdict[a]["rate_span"], 4),
                       share_span=round(verdict[a]["share_span"], 4))
        rec.result(arm="_verdict", **{k: v for k, v in verdict.items() if isinstance(v, bool)})

    print(f"\n=== A3b: {stem} — does d_stance change the KIND of refusal? ===")
    print(f"baseline (c=0): refusal {base['refusal_rate']:.3f} | "
          f"inability {base['n_inability']} identity {base['n_identity']} "
          f"| share {base['inability_share']:.3f} | degenerate {base['degenerate']:.3f}\n")
    print(f"{'arm':8s} {'c':>7s} {'KL':>6s} {'degen':>6s} {'refusal':>8s} {'Δrate':>7s} "
          f"{'inab':>5s} {'iden':>5s} {'share':>7s} {'Δshare':>8s}")
    for name in arms:
        if not results["arms"][name]:
            print(f"{name:8s} no coefficient of either sign is within the KL bound")
            continue
        for c, r in results["arms"][name].items():
            print(f"{name:8s} {float(c):>+7.3f} {r['kl']:>6.3f} {r['degenerate']:>6.3f} "
                  f"{r['refusal_rate']:>8.3f} {r['d_rate']:>+7.3f} {r['n_inability']:>5d} "
                  f"{r['n_identity']:>5d} {r['inability_share']:>7.3f} {r['d_share']:>+8.3f}"
                  + ("" if r["usable"] else "  UNUSABLE"))
    print("\nspans over usable cells (baseline included):")
    for a in arms:
        print(f"  {a:8s} rate {verdict[a]['rate_span']:.3f}   "
              f"composition {verdict[a]['share_span']:.3f}   "
              f"({verdict[a]['n_usable']} usable cell(s))")
    print(f"\n  stance moves composition beyond null : "
          f"{'YES' if verdict['stance_moves_composition'] else 'NO'}")
    print(f"  stance leaves the rate alone (±{RATE_TOLERANCE:.2f}): "
          f"{'YES' if verdict['stance_preserves_rate'] else 'NO'}")
    print(f"  arditi moves the rate (positive control): "
          f"{'YES' if verdict['arditi_moves_rate'] else 'NO'}")
    print(f"\n  DISSOCIATION: {'YES' if verdict['dissociation'] else 'NO'}")
    if verdict["dissociation"]:
        print("  -> d_stance changes WHICH refusal, d_arditi changes HOW MUCH. The stance")
        print("     axis is causal and separable: refusal is multi-directional.")
    elif not verdict["powered"]:
        print("  -> UNDERPOWERED, not negative. No usable cell in one of the two arms that")
        print("     the comparison needs, so nothing here is evidence either way. Widen or")
        print("     lower --grid, or accept that this model cannot be steered in regime.")
    elif not verdict["arditi_moves_rate"]:
        print("  -> POSITIVE CONTROL FAILED inside the regime. The refusal direction itself")
        print("     does not move the rate at any coefficient the model survives, so the")
        print("     stance arm's result is uninterpretable. That is a fact about the harness,")
        print("     not about stance.")
    else:
        print("  -> the falsifier fired: d_stance does not causally control stance while")
        print("     d_arditi does control rate. A3's geometry is consistent with a")
        print("     prompt-content direction, and multi-directionality does not survive.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
