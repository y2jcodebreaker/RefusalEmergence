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

Output: results/{lineage}_{stage}_stance_directions.npz
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


def label_prompts(stem: str, arm: str) -> tuple[list[str], list[str]]:
    """(stance per prompt, completions). Reads the stored behavioural arm."""
    path = f"results/{stem}_refusal_gen128.npz"
    if not os.path.exists(path):
        path = f"results/{stem}_refusal.npz"
    if not os.path.exists(path):
        raise SystemExit(f"no stored completions for {stem}; run run_stage --behavioral first")
    z = np.load(path, allow_pickle=True)
    comps = json.loads(str(z["sample_completions"]))[arm]
    labs = []
    for c in comps:
        t = truncate_at_turn(c)
        labs.append("confusion" if any(x in t.lower() for x in CONFUSION_SUBSTRINGS)
                    else stance_of(c))
    logger.info("[%s/%s] stance counts: %s", stem, arm,
                {s: labs.count(s) for s in sorted(set(labs))})
    return labs, comps


def fit_direction(acts_pos: np.ndarray, acts_neg: np.ndarray) -> np.ndarray:
    """(n_pos, n_layers, d) mean-diff, identical estimator to get_mean_diff."""
    return acts_pos.mean(0) - acts_neg.mean(0)


def induce_sweep(model, tok, cfg, template, refusal_toks, vec, layer, harmless) -> dict:
    best, at = -float("inf"), None
    for c in COEFFS:
        h = _addition_handles(model, torch.tensor(vec, dtype=torch.float32), coeff=c,
                              layer=layer)
        try:
            v = _mean_refusal(_last_logits(model, tok, harmless, template, cfg.batch_size),
                              refusal_toks)
        finally:
            for x in h:
                x.remove()
        if v > best:
            best, at = v, c
    return {"max": float(best), "at_coeff": float(at)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="tulu2_dpo")
    ap.add_argument("--stage", default="dpo")
    ap.add_argument("--arm", default="baseline", choices=("baseline", "ablated"))
    ap.add_argument("--n-null", type=int, default=10)
    args = ap.parse_args()

    cfg = config_for(args.lineage)
    check_disk(cfg)
    ckpts = dict(cfg.checkpoints)
    if args.stage not in ckpts:
        raise SystemExit(f"unknown stage {args.stage!r}; have {list(ckpts)}")
    stem = f"{cfg.lineage}_{args.stage}"

    labs, _ = label_prompts(stem, args.arm)
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
    out, dirs = {}, {}
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
        # makes in transplant.py, and for the same anti-circularity reason.
        best = {"max": -float("inf")}
        for layer in range(d.shape[1]):
            r = induce_sweep(model, tok, cfg, template, refusal_toks,
                             d[d.shape[0] - 1, layer], layer, harmless)
            if r["max"] > best["max"]:
                best, best_layer = r, layer
        dirs[stance] = d[d.shape[0] - 1, best_layer]
        # Shuffled-label null over the SAME partition sizes.
        # Shuffled-label null: same sizes, same pooled activations, labels randomised. It
        # shares the data's anisotropic geometry, so it is strictly harder than an isotropic
        # random vector.
        nulls = []
        pool = np.concatenate([A[pos], N[neg]])
        for _ in range(args.n_null):
            sh = rng.permutation(len(pool))
            dn = fit_direction(pool[sh[:k]], pool[sh[k:2 * k]])
            nulls.append(induce_sweep(model, tok, cfg, template, refusal_toks,
                                      dn[dn.shape[0] - 1, best_layer], best_layer,
                                      harmless)["max"])
        out[stance] = {"n": len(idx), "fitted": True, "k_balanced": int(k),
                       "layer": int(best_layer), "induce_max": best["max"],
                       "at_coeff": best["at_coeff"],
                       "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
                       "z": float((best["max"] - np.mean(nulls)) / (np.std(nulls) + 1e-9)),
                       "n_null_crossing": int(sum(n >= cfg.induce_threshold for n in nulls))}
        logger.info("[%s] n=%d k=%d L%d induce %+.3f @c%.1f | null %+.3f±%.3f | z=%+.1f",
                    stance, len(idx), k, best_layer, best["max"], best["at_coeff"],
                    np.mean(nulls), np.std(nulls), out[stance]["z"])

    cos = {}
    names = sorted(dirs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = dirs[a], dirs[b]
            cos[f"{a}|{b}"] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    if cos:
        logger.info("pairwise |cos| between stance directions: %s",
                    {k: round(abs(v), 3) for k, v in cos.items()})

    path = f"{cfg.results_dir}/{stem}_stance_directions.npz"
    with RunRecord(EXPERIMENT, "stance_directions.py", cfg=cfg, question=QUESTION,
                   notes=f"arm={args.arm}, stances fitted against compliance, count-balanced, "
                         f"{args.n_null} shuffled-label nulls per stance") as rec:
        np.savez(path, stage=np.array(args.stage), arm=np.array(args.arm),
                 results=np.array(json.dumps(out)), cosines=np.array(json.dumps(cos)),
                 **{f"dir_{s}": v for s, v in dirs.items()})
        for s, r in out.items():
            rec.result(stance=s, **{k: v for k, v in r.items() if k != "fitted"})
    print(f"\n=== A3: {stem}/{args.arm} ===")
    print(f"{'stance':14s} {'n':>4s} {'layer':>6s} {'induce':>8s} {'z':>7s} {'nulls crossing':>15s}")
    for s, r in out.items():
        if not r.get("fitted"):
            print(f"{s:14s} {r['n']:>4d}   not fitted (< {MIN_CLASS})")
            continue
        print(f"{s:14s} {r['n']:>4d} {r['layer']:>6d} {r['induce_max']:>+8.3f} "
              f"{r['z']:>+7.1f} {r['n_null_crossing']:>10d}/{args.n_null}")
    if cos:
        print("\npairwise |cos|: " + ", ".join(f"{k} {abs(v):.3f}" for k, v in cos.items()))
        print("  < 0.30 -> distinct directions, refusal is multi-directional")
        print("  > 0.70 -> one direction with several readouts")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
