"""A3 — is refusal multi-directional? One direction per STANCE, or one direction overall?

    python stance_directions.py --lineage tulu2_dpo --stage dpo      # the decisive case
    python stance_directions.py --lineage olmo2_e7  --stage rlvr

THE QUESTION, AS B1 LEFT IT. Four refusal stances exist across three families: inability
("I cannot"), identity ("As an AI language model, I must emphasize"), condemnation ("I
strongly condemn"), normative ("Bribery is illegal"). Ablating Arditi's direction removes
INABILITY wherever it exists -- OLMo 2 128->0, Tulu-2 78->8 -- while Tulu-2's IDENTITY
refusals go 20->21, entirely untouched.

So the direction is stance-specific. The open question is whether the surviving stances have
directions OF THEIR OWN:

    they do, and non-collinear  -> refusal is MULTI-DIRECTIONAL, and Arditi's single-direction
                                   result describes one mode of several
    they do, but collinear      -> one direction, several readouts
    they do not                 -> the surviving stances are not linearly mediated at the eoi
                                   position, which bounds the linear-representation hypothesis

All three are reportable, which is the point: this experiment does not need a particular
answer to be worth running.

WHY TULU-2 IS THE DECISIVE CASE. It is the only model with enough of two stances at once (78
inability, 20 identity) AND a survivor: identity refusals are not removed by ablation. So
d_identity can be fitted, and there is an independent behavioural fact -- 20->21 -- for it to
explain. OLMo 2 can only supply d_normative, and its normative refusals APPEAR after ablation
(2->64) rather than persisting, which is a different phenomenon.

THE NEGATIVE CLASS IS HARMLESS PROMPTS, NOT COMPLIANCE. The first version contrasted each
stance against the harmful prompts the model COMPLIED with, which fails twice: the compliance
class is tiny on a model that refuses (OLMo 2 baseline has 2 of 132, Tulu-2 baseline 10), and
it makes d_inability a different estimator from Arditi's, so the positive control could not be
run at all. Contrasting against HARMLESS prompts is Arditi's own mean-diff, partitioned by
what the model did -- which means d_inability re-fitted here must reproduce the known
direction, and that is the control.

THE CIRCULARITY, NAMED. The stance classes come from completions the model itself produced,
so a direction fitted on them is fitted on the model's own output distribution. Two guards:
a shuffled-label null over the same partition, and a cross-checkpoint transplant (does
d_identity from Tulu-2 induce in OLMo 2?) which is behavioural and immune to the objection.

WHAT A1 ALMOST COST US. The obvious contrast for a non-inability direction is "baseline
third-person refusals vs baseline compliances". Baselines contain ~0 third-person refusals, so
that positive class is EMPTY and the fit would have returned a direction from noise with
nothing flagging it. The contrast must be drawn where the stance actually occurs.

Output: results/{lineage}_{stage}_stance_directions.npz  (v2 run: ..._v2.npz)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import numpy as np
import torch

from config import config_for
from data import load_instructions
from probes import cache_activations
from refusal_direction import (_addition_handles, _last_logits, _mean_refusal,
                               resolve_refusal_token)
from refusal_substring import (CONFUSION_SUBSTRINGS, is_refusal_strict, truncate_at_turn)
from run_stage import load_model, set_seed
from runlog import RunRecord
from transplant import COEFFS
from verify_setup import check_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("A3")

EXPERIMENT = "A3"
QUESTION = ("Do the refusal stances that survive ablation have linear directions of their own, "
            "and are those directions distinct from Arditi's?")

# The four stances, as B1 left them. Order matters: inability is tested first because it is the
# POSITIVE CONTROL -- re-fitting it this way must reproduce the known direction, or the whole
# contrast construction is broken and nothing downstream is readable.
STANCES: tuple[tuple[str, str], ...] = (
    ("inability", r"^\W*(i'?m sorry|i am sorry|i apolog|i cannot|i can'?t|i am unable"
                  r"|i'?m unable|i won'?t|i will not|i do not provide)"),
    ("identity", r"^\W*(as an ai|as a language model|as an assistant)"),
    ("condemnation", r"^\W*(i strongly (condemn|advise|discourage)|i do not (condone|support"
                     r"|encourage)|i must (emphasi|clarif|stress))"),
)
_COMPILED = [(n, re.compile(p, re.I)) for n, p in STANCES]
MIN_CLASS = 12          # below this a mean-diff is noise; report and skip rather than fit


def stance_of(text: str) -> str:
    t = truncate_at_turn(text).strip()
    for name, rx in _COMPILED:
        if rx.match(t):
            return name
    return "normative" if is_refusal_strict(t) or _looks_normative(t) else "compliance"


_NORM = re.compile(r"\b(is|are|can be|would be)\b[^.]{0,40}?\b(illegal|unethical|harmful"
                   r"|dangerous|a crime|not appropriate|not advisable|a violation)", re.I)


def _looks_normative(t: str) -> bool:
    return bool(_NORM.search(t[:200]))


def labeller_for(name: str):
    """v1 or v2, chosen explicitly. v1 is the DEFAULT so every pre-2026-09-26 result
    reproduces untouched; v2 is what the A3 rerun uses (P1 plan section 18).

    The difference is not cosmetic: v1 assigns a register from the completion's OPENING, so
    "I do not condone X. However, here are the steps: 1. ..." counts as a condemnation
    refusal. v2 reads the whole completion. On Tulu-2's baseline arm that moves 20 of 41
    identity items into compliance -- and A3's treatment classes AND its outcome measure are
    both built from these labels."""
    if name == "v1":
        return stance_of
    if name == "v2":
        from stance_v2 import stance_of_v2
        return stance_of_v2
    raise SystemExit(f"unknown labeller {name!r}; use v1 or v2")


def label_prompts(stem: str, arm: str, labeller=None) -> tuple[list[str], list[str]]:
    """(stance per prompt, completions). Reads the stored behavioural arm."""
    path = f"results/{stem}_refusal_gen128.npz"
    if not os.path.exists(path):
        path = f"results/{stem}_refusal.npz"
    if not os.path.exists(path):
        raise SystemExit(f"no stored completions for {stem}; run run_stage --behavioral first")
    z = np.load(path, allow_pickle=True)
    comps = json.loads(str(z["sample_completions"]))[arm]
    lab_fn = labeller or stance_of
    labs = []
    for c in comps:
        t = truncate_at_turn(c)
        labs.append("confusion" if any(x in t.lower() for x in CONFUSION_SUBSTRINGS)
                    else lab_fn(c))
    logger.info("[%s/%s] stance counts: %s", stem, arm,
                {s: labs.count(s) for s in sorted(set(labs))})
    return labs, comps


def fit_direction(acts_pos: np.ndarray, acts_neg: np.ndarray) -> np.ndarray:
    """(n_pos, n_layers, d) mean-diff, identical estimator to get_mean_diff."""
    return acts_pos.mean(0) - acts_neg.mean(0)


def _induce_at(model, tok, cfg, template, refusal_toks, vec, layer, harmless,
               coeff: float) -> float:
    """Mean refusal-token logit on HARMLESS prompts after adding `coeff * vec` at `layer`."""
    h = _addition_handles(model, torch.tensor(vec, dtype=torch.float32), coeff=coeff,
                          layer=layer)
    try:
        return float(_mean_refusal(
            _last_logits(model, tok, harmless, template, cfg.batch_size), refusal_toks))
    finally:
        for x in h:
            x.remove()


def select_cell(model, tok, cfg, template, refusal_toks, d, harmless) -> tuple:
    """(pos*, layer*, surface) by induce at coeff=1.0 over the FULL (pos, layer) surface.

    THIS SWEEPS POSITIONS, AND THE FIRST VERSION DID NOT. It fixed the eoi position to the
    last one -- `d[d.shape[0] - 1, layer]` -- and swept layers only. That is not a cheap
    approximation of Arditi's selection, it is a different selection: on tulu-2-dpo the
    model's own (pos*, l*) is (2, 14) of a 5-position window, so the fixed-position sweep
    could not reach the model's own cell and landed on (4, 12) instead. The positive control
    (does a re-fitted d_inability reproduce the known direction?) therefore could not pass,
    and the 2026-09-22 run reported a cosine whose cells nothing else in the project used.

    COEFF 1.0, NOT A COEFFICIENT SWEEP, BECAUSE THAT IS WHAT run_stage STORES. The `steer`
    surface in {stem}_refusal.npz is refusal_strength_curve's addition at coeff=1.0 across
    every (pos, layer). Selecting here on the same quantity is what makes the two numbers
    comparable at all; the earlier per-layer max over six coefficients was not the same
    measurement, so even the layers it did visit could not be read against run_stage. The
    coefficient sweep still happens -- once, at the selected cell -- which is also cheaper
    than the old sweep (n_pos*n_layers + |COEFFS| evaluations, not n_layers*|COEFFS|)."""
    n_pos, n_layers = d.shape[0], d.shape[1]
    surf = np.full((n_pos, n_layers), np.nan)
    for pos in range(n_pos):
        for layer in range(n_layers):
            surf[pos, layer] = _induce_at(model, tok, cfg, template, refusal_toks,
                                          d[pos, layer], layer, harmless, 1.0)
    p, l = (int(v) for v in np.unravel_index(np.nanargmax(surf), surf.shape))
    return p, l, surf


def induce_sweep(model, tok, cfg, template, refusal_toks, vec, layer, harmless) -> dict:
    """Best induce over COEFFS at ONE cell. Reported as the stance's effect size."""
    best, at = -float("inf"), None
    for c in COEFFS:
        v = _induce_at(model, tok, cfg, template, refusal_toks, vec, layer, harmless, c)
        if v > best:
            best, at = v, c
    return {"max": float(best), "at_coeff": float(at)}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def halves(idx, rng) -> tuple:
    p = rng.permutation(np.asarray(idx))
    return p[: len(p) // 2], p[len(p) // 2:]


def paired_cos(pool_a, pool_b, neg, rng, pos: int, layer: int, k: int,
               n_rep: int = 20) -> tuple:
    """mean |cos| between two mean-diffs, each = mean(k from its pool) - mean(k from neg).

    Draws from `neg` are INDEPENDENT and overlapping, exactly as the real per-stance fits
    draw them, so whatever inflation the shared harmless class contributes is present on
    both sides of every comparison and cancels when one is read against the other.

    Used two ways, and they only mean anything together:
      CEILING  pool_a, pool_b = disjoint halves of the SAME stance -> what |cos| looks like
               when the two directions ARE the same direction, at this n.
      OBSERVED pool_a, pool_b = two different stances.
    Both at the same k, because a cosine's noise floor depends on n. The 2026-09-22 run
    reported a full-n between-stance cosine (0.972, n=78 vs n=41) against no ceiling at all,
    and a half-n ceiling would have been the wrong comparator in the direction that flatters
    the interesting answer."""
    out = []
    for _ in range(n_rep):
        a = (pool_a[rng.choice(len(pool_a), k, replace=False)].mean(0)
             - neg[rng.choice(len(neg), k, replace=False)].mean(0))
        b = (pool_b[rng.choice(len(pool_b), k, replace=False)].mean(0)
             - neg[rng.choice(len(neg), k, replace=False)].mean(0))
        out.append(abs(_cos(a[pos, layer], b[pos, layer])))
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="tulu2_dpo")
    ap.add_argument("--stage", default="dpo")
    ap.add_argument("--arm", default="baseline", choices=("baseline", "ablated"))
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--labels", default="v1", choices=("v1", "v2"),
                    help="which stance labeller defines the classes. v1 reproduces the "
                         "2026-09-22 result; v2 is the rerun that retired it (P1 plan 18). "
                         "Outputs are suffixed so the two never overwrite each other.")
    args = ap.parse_args()

    cfg = config_for(args.lineage)
    if not check_disk(cfg, stages=(args.stage,)):
        raise SystemExit("free disk (or set HF_HOME) before loading weights.")
    ckpts = dict(cfg.checkpoints)
    if args.stage not in ckpts:
        raise SystemExit(f"unknown stage {args.stage!r}; have {list(ckpts)}")
    stem = f"{cfg.lineage}_{args.stage}"

    labs, _ = label_prompts(stem, args.arm, labeller_for(args.labels))
    tail = load_instructions("harmful_train")[cfg.n_train:]
    prompts = tail[: cfg.n_behavioral] if cfg.n_behavioral else tail
    if len(prompts) != len(labs):
        raise SystemExit(f"{len(labs)} completions but {len(prompts)} prompts -- the stored arm "
                         f"and the split disagree; check n_train/n_behavioral for this lineage")

    groups = {s: [i for i, x in enumerate(labs) if x == s] for s in set(labs)}
    logger.info("fittable stances (>= %d): %s", MIN_CLASS,
                [s for s in groups if len(groups[s]) >= MIN_CLASS and s != "compliance"])

    set_seed(cfg.seed)
    model, tok = load_model(ckpts[args.stage], cfg.dtype)
    template, want_id, n_eoi, _ = cfg.regime(args.stage)
    refusal_toks = [resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)]
    harmless = load_instructions("harmless_val")[: cfg.n_val]

    # Negative class: HARMLESS prompts, the same ones Arditi's estimator uses. Held at the
    # size of the largest stance class so every fit can be count-balanced.
    n_neg = max(len(v) for v in groups.values())
    harmless_fit = load_instructions("harmless_train")[: max(n_neg, MIN_CLASS)]
    logger.info("caching activations: %d harmful prompts + %d harmless",
                len(prompts), len(harmless_fit))
    A = cache_activations(model, tok, prompts, template, n_eoi, cfg.batch_size).numpy()
    N = cache_activations(model, tok, harmless_fit, template, n_eoi, cfg.batch_size).numpy()

    rng = np.random.default_rng(cfg.seed)
    out, dirs, surfaces, fitted = {}, {}, {}, {}
    for stance in [s for s, _ in STANCES] + ["normative"]:
        idx = groups.get(stance, [])
        if len(idx) < MIN_CLASS:
            logger.info("[%s] %d examples < %d -- not fitted", stance, len(idx), MIN_CLASS)
            out[stance] = {"n": len(idx), "fitted": False}
            continue
        # Count-balanced: an unbalanced mean-diff is dominated by the larger class's
        # idiosyncrasies, and the stance classes differ by up to 6x here.
        k = min(len(idx), len(N))
        pos = rng.choice(idx, k, replace=False)
        neg = rng.choice(len(N), k, replace=False)
        d = fit_direction(A[pos], N[neg])                       # (n_pos, n_layers, dim)
        # Pick the cell by induce, the axis being scored -- the same choice --source-by induce
        # makes in transplant.py, and for the same anti-circularity reason. Position AND
        # layer: see select_cell for what fixing the position cost.
        p_star, best_layer, surf = select_cell(model, tok, cfg, template, refusal_toks,
                                               d, harmless)
        best = induce_sweep(model, tok, cfg, template, refusal_toks,
                            d[p_star, best_layer], best_layer, harmless)
        dirs[stance] = d[p_star, best_layer]
        surfaces[stance] = surf
        fitted[stance] = d          # kept so the control can read OTHER cells
        # Shuffled-label null: same sizes, same pooled activations, labels randomised. It
        # shares the data's anisotropic geometry, so it is strictly harder than an isotropic
        # random vector.
        #
        # The null is evaluated AT THE CELL THE REAL DIRECTION SELECTED, not at its own best
        # cell -- it is not given the same n_pos*n_layers search. So z answers "at this cell,
        # is the effect label-driven?", which is the question, and NOT "is this cell special?",
        # which it would overstate. Giving each null its own surface would cost n_null times
        # the sweep; the asymmetry is stated here rather than silently priced in.
        nulls = []
        pool = np.concatenate([A[pos], N[neg]])
        for _ in range(args.n_null):
            sh = rng.permutation(len(pool))
            dn = fit_direction(pool[sh[:k]], pool[sh[k:2 * k]])
            nulls.append(induce_sweep(model, tok, cfg, template, refusal_toks,
                                      dn[p_star, best_layer], best_layer,
                                      harmless)["max"])
        out[stance] = {"n": len(idx), "fitted": True, "k_balanced": int(k),
                       "layer": int(best_layer), "pos": int(p_star),
                       "induce_max": best["max"], "at_coeff": best["at_coeff"],
                       "induce_at_c1": float(surf[p_star, best_layer]),
                       "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
                       "z": float((best["max"] - np.mean(nulls)) / (np.std(nulls) + 1e-9)),
                       "n_null_crossing": int(sum(n >= cfg.induce_threshold for n in nulls))}
        logger.info("[%s] n=%d k=%d pos%d/L%d induce %+.3f @c%.1f (c1 %+.3f) | "
                    "null %+.3f±%.3f | z=%+.1f",
                    stance, len(idx), k, p_star, best_layer, best["max"], best["at_coeff"],
                    surf[p_star, best_layer], np.mean(nulls), np.std(nulls), out[stance]["z"])

    # ------------------------------------------------------------- positive control
    # d_inability re-fitted here must reproduce the direction run_stage already found, or
    # the contrast construction is broken and every cosine below is unreadable. Two halves,
    # both pre-registered before this run:
    #   CELL      A3's selected (pos*, l*) equals the argmax of run_stage's stored `steer`
    #   DIRECTION cos(d_inability, d_arditi) >= 0.70 at that cell
    #
    # THE COMPARATOR IS steer's ARGMAX, NOT THE STORED (pos_star, l_star). Those two are not
    # the same cell and were never meant to be: run_stage selects by ABLATION under Arditi's
    # three criteria, A3 selects by INDUCE. On tulu-2-dpo, Arditi's OWN direction induces
    # +0.826 somewhere on the surface and only +0.519 at its ablation-selected (2, 14) -- so
    # demanding that A3 land on (2, 14) would fail a direction that is exactly right, for a
    # reason that has nothing to do with the stance partition. Same rule, same quantity,
    # different fit: that is what makes the comparison diagnostic.
    #
    # d_arditi is recomputed from the SAME cached activations, unpartitioned -- literally
    # Arditi's estimator, on an independent split (the held-out tail, not harmful_train[:128]),
    # which makes the agreement a stronger check than re-running it on the fitting split.
    ctrl: dict = {"checked": False}
    ref_path = cfg.path(args.stage, "refusal")
    if "inability" not in dirs:
        ctrl["why"] = "inability not fitted -- there is no positive control"
    elif not os.path.exists(ref_path):
        ctrl["why"] = f"{ref_path} missing -- run run_stage first"
    else:
        z = np.load(ref_path, allow_pickle=True)
        steer = np.asarray(z["steer"], dtype=float) if "steer" in z.files else None
        if steer is None or not np.isfinite(steer).any():
            ctrl["why"] = ("run_stage stored no usable steer surface for this stage "
                           "(it was run with filtered=False); re-run it with the KL filter")
        elif steer.shape != fitted["inability"].shape[:2]:
            ctrl["why"] = (f"steer surface is {steer.shape} but A3 cached "
                           f"{fitted['inability'].shape[:2]} -- n_eoi or the layer count "
                           f"changed between runs, so the cells are not the same cells")
        else:
            ref_pos, ref_l = (int(v) for v in
                              np.unravel_index(np.nanargmax(steer), steer.shape))
            got_pos, got_l = out["inability"]["pos"], out["inability"]["layer"]
            # The SAME count-balanced d_inability the experiment uses, read at run_stage's
            # cell -- not a second, differently-sampled fit, which would leave the control
            # certifying a direction no result depends on.
            va = fitted["inability"][ref_pos, ref_l]
            vb = (A.mean(0) - N.mean(0))[ref_pos, ref_l]
            c = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
            ctrl = {"checked": True,
                    "ref_pos": ref_pos, "ref_layer": ref_l,
                    "ref_steer_max": float(np.nanmax(steer)),
                    "ablation_pos_star": int(z["pos_star"]), "ablation_l_star": int(z["l_star"]),
                    "got_pos": got_pos, "got_layer": got_l,
                    "cell_match": bool(got_pos == ref_pos and got_l == ref_l),
                    "cos_with_arditi": c, "direction_match": bool(c >= 0.70)}
            ctrl["passed"] = bool(ctrl["cell_match"] and ctrl["direction_match"])

    # -------------------------------------------------------------- geometry, read properly
    # THE CONTRAST CANNOT SEPARATE STANCE FROM HARMFULNESS, AND THE FIRST VERSION DID NOT SAY
    # SO. Every stance direction here is mean(harmful & stance) - mean(harmless): the stance
    # label chooses WHICH harmful prompts enter the positive class, but the axis being fitted
    # is harmful-vs-harmless either way. Two such fits share a subtrahend and a dominant
    # signal, so a high pairwise cosine is close to guaranteed by construction and says
    # nothing about stance. The 2026-09-22 run reported 0.972 with nothing to read it against.
    # Two additions make it readable, and neither needs another forward pass over the corpus:
    #
    #   CEILING  |cos| between two fits of the SAME stance on disjoint halves -- what
    #            agreement looks like when the direction is identical, at this n.
    #   CONTRAST mean(inability) - mean(identity): both classes harmful, both refused,
    #            differing only in the stance rendered. The only within-harmful test of
    #            whether stance has a linear axis, and its cosine against d_arditi is the
    #            multi-directionality number this experiment actually wants.
    #
    # It also has to reconcile with B1: ablating d_arditi takes inability 78->8 and leaves
    # identity 20->21 UNTOUCHED. A d_identity genuinely collinear with d_arditi could not do
    # that, so either the pairwise cosine is an artifact of the shared contrast or identity
    # refusal is not linearly mediated at the eoi position. Those are different claims and
    # the ceiling is what separates them.
    geo: dict = {}
    if ctrl.get("checked"):
        c_pos, c_lay = ctrl["ref_pos"], ctrl["ref_layer"]
    elif dirs:
        c_pos, c_lay = (lambda f: (out[f]["pos"], out[f]["layer"]))(next(iter(dirs)))
    else:
        c_pos = c_lay = None

    stance_names = sorted(dirs)
    if c_pos is not None and len(stance_names) >= 2:
        geo["cell"] = [int(c_pos), int(c_lay)]
        d_arditi_cell = (A.mean(0) - N.mean(0))[c_pos, c_lay]
        # One k for the whole block, so ceiling and observed are the same measurement at the
        # same sample size: half the SMALLEST stance, since the ceiling needs two disjoint
        # halves of it.
        k_geo = min(len(groups[x]) for x in stance_names) // 2
        geo["k_matched"] = int(k_geo)
        if k_geo < 8:
            geo["why_no_ceiling"] = f"smallest stance gives k={k_geo}; too few to split"
        else:
            for x in stance_names:
                h1, h2 = halves(groups[x], rng)
                m, sd = paired_cos(A[h1], A[h2], N, rng, c_pos, c_lay, k_geo)
                geo[f"ceiling_{x}"] = {"mean": m, "sd": sd}
            for i, a in enumerate(stance_names):
                for b in stance_names[i + 1:]:
                    m, sd = paired_cos(A[groups[a]], A[groups[b]], N, rng,
                                       c_pos, c_lay, k_geo)
                    geo[f"observed_{a}|{b}"] = {"mean": m, "sd": sd}

        if "inability" in dirs and "identity" in dirs and k_geo >= 8:
            ia, idn = np.asarray(groups["inability"]), np.asarray(groups["identity"])
            kk = min(len(ia), len(idn))
            d_stance = (A[rng.choice(ia, kk, replace=False)].mean(0)
                        - A[rng.choice(idn, kk, replace=False)].mean(0))
            a1, a2 = halves(ia, rng)
            b1, b2 = halves(idn, rng)
            rel = []
            for _ in range(20):
                d1 = (A[rng.choice(a1, k_geo, replace=False)].mean(0)
                      - A[rng.choice(b1, k_geo, replace=False)].mean(0))
                d2 = (A[rng.choice(a2, k_geo, replace=False)].mean(0)
                      - A[rng.choice(b2, k_geo, replace=False)].mean(0))
                rel.append(abs(_cos(d1[c_pos, c_lay], d2[c_pos, c_lay])))
            # Null: pool the two stances, split into two PSEUDO-stances at random, ask the
            # same reliability question. A real stance axis has to beat this.
            pooled = np.concatenate([ia, idn])
            nrel = []
            for _ in range(args.n_null):
                sh = rng.permutation(pooled)
                qa1, qa2 = halves(sh[: len(ia)], rng)
                qb1, qb2 = halves(sh[len(ia):], rng)
                if min(len(qa1), len(qa2), len(qb1), len(qb2)) < k_geo:
                    continue
                d1 = (A[rng.choice(qa1, k_geo, replace=False)].mean(0)
                      - A[rng.choice(qb1, k_geo, replace=False)].mean(0))
                d2 = (A[rng.choice(qa2, k_geo, replace=False)].mean(0)
                      - A[rng.choice(qb2, k_geo, replace=False)].mean(0))
                nrel.append(abs(_cos(d1[c_pos, c_lay], d2[c_pos, c_lay])))
            sw = induce_sweep(model, tok, cfg, template, refusal_toks,
                              d_stance[c_pos, c_lay], c_lay, harmless)
            geo["stance_contrast"] = {
                "k_balanced": int(kk), "k_matched": int(k_geo),
                "reliability": {"mean": float(np.mean(rel)), "sd": float(np.std(rel))},
                "null_reliability": {"mean": float(np.mean(nrel)) if nrel else float("nan"),
                                     "sd": float(np.std(nrel)) if nrel else float("nan")},
                "cos_with_arditi": _cos(d_stance[c_pos, c_lay], d_arditi_cell),
                "cos_with_inability": _cos(d_stance[c_pos, c_lay],
                                           fitted["inability"][c_pos, c_lay]),
                "induce_max": sw["max"], "at_coeff": sw["at_coeff"]}
            dirs["_stance_contrast"] = d_stance[c_pos, c_lay]

    cos = {}
    names = sorted(k for k in dirs if not k.startswith("_"))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = dirs[a], dirs[b]
            cos[f"{a}|{b}"] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    if cos:
        logger.info("pairwise |cos| between stance directions: %s",
                    {k: round(abs(v), 3) for k, v in cos.items()})

    suffix = "" if args.labels == "v1" else f"_{args.labels}"
    path = f"{cfg.results_dir}/{stem}_stance_directions{suffix}.npz"
    with RunRecord(EXPERIMENT, "stance_directions.py", cfg=cfg, question=QUESTION,
                   notes=f"labels={args.labels}, arm={args.arm}, stances fitted against "
                         f"HARMLESS (not compliance), "
                         f"count-balanced, cell by induce@c1.0 over the full (pos, layer) "
                         f"surface, {args.n_null} shuffled-label nulls per stance; "
                         f"positive control passed={ctrl.get('passed')}") as rec:
        np.savez(path, stage=np.array(args.stage), arm=np.array(args.arm),
                 results=np.array(json.dumps(out)), cosines=np.array(json.dumps(cos)),
                 control=np.array(json.dumps(ctrl)),
                 geometry=np.array(json.dumps(geo)),
                 **{f"dir_{s}": v for s, v in dirs.items()},
                 **{f"surface_{s}": v for s, v in surfaces.items()})
        for s, r in out.items():
            rec.result(stance=s, **{k: v for k, v in r.items() if k != "fitted"})
        rec.result(stance="_positive_control", **ctrl)
    print(f"\n=== A3: {stem}/{args.arm} ===")
    print(f"{'stance':14s} {'n':>4s} {'layer':>6s} {'induce':>8s} {'z':>7s} {'nulls crossing':>15s}")
    for s, r in out.items():
        if not r.get("fitted"):
            print(f"{s:14s} {r['n']:>4d}   not fitted (< {MIN_CLASS})")
            continue
        print(f"{s:14s} {r['n']:>4d} {r['layer']:>6d} {r['induce_max']:>+8.3f} "
              f"{r['z']:>+7.1f} {r['n_null_crossing']:>10d}/{args.n_null}")
    print("\n-- positive control (d_inability must reproduce Arditi's direction) --")
    if not ctrl.get("checked"):
        print(f"   NOT CHECKED: {ctrl.get('why', 'unavailable')}")
        print("   Nothing below is reportable without this.")
    else:
        print(f"   cell      {'PASS' if ctrl['cell_match'] else 'FAIL'}  "
              f"A3 picked (pos {ctrl['got_pos']}, L{ctrl['got_layer']}); run_stage's steer "
              f"argmax is (pos {ctrl['ref_pos']}, L{ctrl['ref_layer']})")
        print(f"   direction {'PASS' if ctrl['direction_match'] else 'FAIL'}  "
              f"cos(d_inability, d_arditi) = {ctrl['cos_with_arditi']:+.3f} there "
              f"(pre-registered >= 0.70)")
        print(f"   (for reference, run_stage's ABLATION cell is "
              f"(pos {ctrl['ablation_pos_star']}, L{ctrl['ablation_l_star']}) -- a different "
              f"criterion, not the comparator)")
        if not ctrl["passed"]:
            print("   -> the contrast construction does NOT reproduce the known direction.")
            print("      The cosines below are NOT reportable until this passes.")
    if cos:
        print("\npairwise |cos| (stance-vs-harmless fits): "
              + ", ".join(f"{k} {abs(v):.3f}" for k, v in cos.items()))
        print("  NOT readable alone -- both fits are harmful-vs-harmless, with the stance")
        print("  label only choosing which harmful prompts go in. See the ceiling below.")

    if geo.get("cell"):
        print(f"\n-- geometry at the canonical cell (pos {geo['cell'][0]}, "
              f"L{geo['cell'][1]}), every fit at k={geo.get('k_matched')} --")
        if geo.get("why_no_ceiling"):
            print(f"   no ceiling: {geo['why_no_ceiling']}")
        else:
            print("   ceiling = |cos| between two fits of the SAME stance on disjoint halves;")
            print("   that is what agreement looks like when the direction is identical.")
            for key in sorted(x for x in geo if x.startswith("ceiling_")):
                r = geo[key]
                print(f"     {key[len('ceiling_'):]:14s} {r['mean']:.3f} ± {r['sd']:.3f}")
            for key in sorted(x for x in geo if x.startswith("observed_")):
                pair = key[len("observed_"):]
                a, b = pair.split("|")
                r, ca, cb = geo[key], geo.get(f"ceiling_{a}"), geo.get(f"ceiling_{b}")
                ceil = min(ca["mean"], cb["mean"]) if ca and cb else float("nan")
                verdict = ("AT the ceiling -- not resolvably different directions"
                           if r["mean"] >= ceil - max(r["sd"], 0.01)
                           else "BELOW the ceiling -- a difference survives the noise")
                print(f"     {pair}: observed {r['mean']:.3f} ± {r['sd']:.3f} vs "
                      f"ceiling {ceil:.3f}\n       -> {verdict}")

        sc = geo.get("stance_contrast")
        if sc:
            print("\n   stance contrast: mean(inability) - mean(identity), both classes")
            print("   harmful and refused -- the only within-harmful test of a stance axis.")
            print(f"     reliability           {sc['reliability']['mean']:.3f} ± "
                  f"{sc['reliability']['sd']:.3f}")
            print(f"     pseudo-stance null    {sc['null_reliability']['mean']:.3f} ± "
                  f"{sc['null_reliability']['sd']:.3f}")
            print(f"     |cos| vs d_arditi     {abs(sc['cos_with_arditi']):.3f}")
            print(f"     |cos| vs d_inability  {abs(sc['cos_with_inability']):.3f}")
            print(f"     induces refusal       {sc['induce_max']:+.3f} @c{sc['at_coeff']:.1f}")
            print("\n   read it as:")
            print("     reliability at the null   -> NO stance axis; the surviving stance is")
            print("                                  not linearly mediated at the eoi position.")
            print("                                  Bounds the linear-representation")
            print("                                  hypothesis -- the reportable negative.")
            print("     reliable and |cos| < 0.30 -> a SEPARATE stance axis: multi-directional.")
            print("     reliable and |cos| > 0.70 -> stance rides the refusal axis itself.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
