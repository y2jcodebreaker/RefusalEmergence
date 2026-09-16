"""P1-E1b + P1-E2 — Does the ALIGNED model's refusal direction induce refusal in BASE?

    python transplant.py --stage all

Why this and not cosine. P1-E1 established the harmful/harmless distinction is readable in
base (~0.99 from L1) but could not establish whether it lies on the SAME axis the aligned
model refuses along: SFT and DPO are fine-tunes of base, so their activation geometries are
nearly identical and any two matched noise directions already align. The shuffled-label null
came back at median 0.42-0.45 (0.94 at L1), leaving cosine with almost no power.

A transplant answers the question behaviourally, so anisotropy cannot touch it:

    take DPO's validated refusal direction (its l*), ADD it to the BASE model at the same
    layer, and ask whether base starts refusing HARMLESS prompts.

  induces refusal in base -> base already had the machinery; alignment found and amplified
                             an existing direction (H, strong form)
  does nothing in base    -> the direction only functions inside the aligned model's
                             circuitry; alignment built the COUPLING (H, coupling form)

Folds in P1-E2: every cell is swept over coefficients, which retires the standing objection
that E02 only tested induction at Arditi's default coeff=1.0. KL on harmless prompts is tracked
and REPORTED, but deliberately NOT used as a gate on induction.

That last point cost a run (2026-09-13). Arditi's KL <= 0.1 bound belongs to ABLATION: a
small subtractive perturbation that must leave harmless behaviour untouched. ADDITION is the
opposite -- its whole purpose is to change behaviour on harmless prompts. Gating induction on
"the model barely changed" is self-contradictory, and it reported "no induction" for every
cell including the sanity check (dpo's own direction in dpo, which E02 measured at +1.08).
E02 had this right: its KL filter gated ablation only. Coherence is instead judged from the
sweep SHAPE -- induced refusal rises, peaks, then collapses negative as the model breaks --
and from generations at the best coefficient.

Transplanting across checkpoints is legitimate HERE only because they share one architecture
and lineage; fine-tuning barely rotates the residual basis. A norm-matched random direction
is swept in every cell as the control.

Output: results/{stage}_transplant.npz     Ledger: results/RUNLOG.md
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

from config import config_for
from data import assert_available, load_instructions
from refusal_direction import (_addition_handles, _last_logits, _mean_refusal, kl_last,
                               resolve_refusal_token)
from run_stage import load_model, set_seed
from runlog import RunRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("P1-E1b")

EXPERIMENT = "P1-E1b"
QUESTION = ("Does the aligned model's refusal direction induce refusal when transplanted "
            "into BASE, and at what coefficient does induction appear while KL stays sane?")

COEFFS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)          # RAW mode (Arditi default)

# UNIT-NORM mode. A fixed grid in absolute injected norm is only meaningful if it spans the
# scale the directions actually live at, and that scale is a property of the FAMILY: Zephyr's
# source norms are 1.1 / 7.4 / 4.4, OLMo 2's are an order of magnitude larger. With the fixed
# grid above, OLMo 2's whole matrix sat BELOW the raw operating point -- every cell read "no
# induction", including rlvr->rlvr, which run_stage had already measured as inducing (the
# induce filter is what let L24 be selected at all). Nine "no"s that look like a finding and
# are actually an under-powered sweep: the O-50 failure mode, second occurrence.
#
# So the grid is anchored to the data: multiples of the LARGEST source norm in the lineage.
# One anchor shared by every source keeps injected norm matched across sources (the whole
# point of --unit-norm) while guaranteeing each source's own raw scale falls inside the sweep.
NORM_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def coeff_grid(raw_norms: dict, unit_norm: bool) -> tuple[float, ...]:
    """Coefficients to sweep. Raw mode: Arditi's fixed grid. Unit-norm mode: multiples of the
    largest source norm, so `coeff` reads as injected norm and the raw operating point of
    every source is inside the grid."""
    if not unit_norm:
        return COEFFS
    anchor = max(raw_norms.values())
    return tuple(round(m * anchor, 4) for m in NORM_MULTIPLES)


def source_layers(cfg) -> list[tuple[str, int, int]]:
    """(stage, layer, pos_idx) per source, READ FROM each stage's saved sweep — never
    hardcoded. Uses the Arditi-filtered l*; falls back to the unfiltered argmax when no
    direction passed the filters (base), which is flagged in the log because such a layer
    is 'best among allowed', not a validated refusal layer."""
    out = []
    for stage in cfg.stages:
        path = cfg.path(stage, "refusal")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — run run_stage.py --lineage {cfg.lineage} "
                             f"--stage all first (transplant needs each stage's l*)")
        d = np.load(path, allow_pickle=True)
        l_star, pos_star = int(d["l_star"]), int(d["pos_star"])
        if l_star < 0:
            l_star = int(d["naive_l_star"])
            # The STAGE's own window, not the lineage default: a stage on a regime override
            # has a different n_eoi (olmo2 base: 3, not 6), so cfg.n_eoi - 1 indexed past the
            # end of its directions array. IndexError, 2026-09-16.
            pos_star = cfg.regime(stage)[2] - 1
            logger.warning("[%s] no filtered l* — falling back to the unfiltered argmax L%d, "
                           "pos %d. This is NOT a validated refusal layer.",
                           stage, l_star, pos_star)
        out.append((stage, l_star, pos_star))
    return out


def load_source_directions(cfg, sources, unit_norm: bool = False) -> tuple[dict, dict]:
    """Reuse the P1-E1 probe directions: identical estimator, data and hook point as E02's
    refusal directions (mean-diff harmful-harmless at resid_pre over eoi positions)."""
    out, raw_norms = {}, {}
    for stage, layer, pos_idx in sources:
        path = cfg.path(stage, "probe")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — run probe_representation.py --stage all first")
        d = np.load(path, allow_pickle=True)["directions"]
        if not (0 <= pos_idx < d.shape[0] and 0 <= layer < d.shape[1]):
            raise SystemExit(
                f"[{stage}] (pos={pos_idx}, layer={layer}) is outside its directions array "
                f"{d.shape}. A stage on a regime override has its own n_eoi — check "
                f"cfg.regime('{stage}') against the array this probe run produced.")
        v = d[pos_idx, layer].astype(np.float32)
        raw_norms[(stage, layer, pos_idx)] = float(np.linalg.norm(v))
        if unit_norm:
            # Norms differ 1.1 / 7.4 / 4.4 across base/sft/dpo, so a raw coefficient is not
            # comparable across sources. Unit-normalising makes the coefficient the INJECTED
            # NORM, which is. Changes what the numbers mean -- report which mode was used.
            v = v / (np.linalg.norm(v) + 1e-8)
        out[(stage, layer, pos_idx)] = v
    return out, raw_norms


def sweep_cell(model, tok, direction: torch.Tensor, layer: int, harmless, template,
               refusal_toks, base_harmless_logits, batch_size, coeffs=COEFFS):
    """Induced refusal and KL on harmless prompts, across `coeffs`, for ONE direction."""
    rows = []
    for c in coeffs:
        h = _addition_handles(model, direction, coeff=c, layer=layer)
        try:
            lg = _last_logits(model, tok, harmless, template, batch_size)
            rows.append((c, _mean_refusal(lg, refusal_toks), kl_last(base_harmless_logits, lg)))
        finally:
            for x in h:
                x.remove()
    return rows


def positive_control(records, stage: str, threshold: float):
    """(ok, best_induced, self_cells) — did the target's OWN direction induce refusal in it?

    `records` rows are (src, layer, kind, coeff, induced_refusal, kl). The self-cell is the
    positive control for the whole matrix: see the comment at its call site."""
    self_cells = [r for r in records if r[0] == stage and r[2] == "direction"]
    best = max((r[4] for r in self_cells), default=float("nan"))
    return bool(self_cells) and best >= threshold, best, self_cells


def run_one(stage: str, model_id: str, cfg, srcs, rec: RunRecord, coeffs=COEFFS) -> None:
    set_seed(cfg.seed)
    model, tok = load_model(model_id, cfg.dtype)
    template, want_id, _n, is_ov = cfg.regime(stage)
    if is_ov:
        logger.warning("[%s] REGIME OVERRIDE: template=%r token=%d — this TARGET is evaluated "
                       "in its own regime; induced-refusal values do not compare across the "
                       "boundary, but 'does it cross threshold at all' does.",
                       stage, template, want_id)
        refusal_toks = [want_id]
    else:
        refusal_toks = [resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)]
    harmless = load_instructions("harmless_val")[: cfg.n_transplant]

    base_lg = _last_logits(model, tok, harmless, template, cfg.batch_size)
    baseline = _mean_refusal(base_lg, refusal_toks)
    logger.info("[%s] baseline refusal on HARMLESS = %.3f over n=%d (induction must beat %.2f)",
                stage, baseline, len(harmless), cfg.induce_threshold)

    gen = torch.Generator().manual_seed(cfg.seed)
    records = []
    for (src, layer, _pos), vec in srcs.items():
        d = torch.from_numpy(vec).to(model.device)
        for kind, direction in (("direction", d),
                                ("random", _norm_matched(d, gen))):
            for c, ref, kl in sweep_cell(model, tok, direction, layer, harmless, template,
                                         refusal_toks, base_lg, cfg.batch_size, coeffs):
                records.append((src, layer, kind, c, ref, kl))
            last = records[-len(coeffs):]      # exactly this cell's coefficient sweep
            best = max(last, key=lambda r: r[4])
            ok = [c for _, _, _, c, r, _ in last if r >= cfg.induce_threshold]
            logger.info("[%s] src=%-4s L%-2d %-9s | max induced %+.3f @coeff %.1f (KL %.2f) | %s",
                        stage, src, layer, kind, best[4], best[3], best[5],
                        f"INDUCES (first at coeff {min(ok)})" if ok
                        else "never crosses threshold")

    # POSITIVE CONTROL. A transplant matrix of all-"no" is only a finding if the sweep was
    # powerful enough to produce a "yes" where one must exist. The self-cell is that test:
    # a stage's OWN direction at its OWN l* passed run_stage's induce filter by construction
    # (select_direction_arditi: valid &= steering_refusal >= induce_threshold), so it MUST
    # cross here too. When it does not, the grid is below the operating point and nothing in
    # the matrix can be interpreted.
    #
    # Exempt: the first stage. Base failing to induce in itself is the measurement, not a
    # malfunction -- base has no filtered l* at all (l* = -1, unfiltered fallback), so there
    # is no guarantee to violate.
    is_first = stage == cfg.stages[0]
    ctrl_ok, self_best, self_cells = positive_control(records, stage, cfg.induce_threshold)
    if not is_first and not ctrl_ok:
        logger.error(
            "[%s] POSITIVE CONTROL FAILED: %s's own direction @L%d does not induce refusal in "
            "%s (best %+.3f < %.2f) — but run_stage's induce filter already established that it does at raw "
            "scale. The sweep tops out at injected norm %.1f, below this direction's raw norm. "
            "DO NOT report this matrix: every 'no' in it is under-powered, not negative.",
            stage, stage, self_cells[0][1], stage, self_best, cfg.induce_threshold,
            max(coeffs))
    elif not is_first:
        logger.info("[%s] positive control OK: own direction induces (best %+.3f)",
                    stage, self_best)

    arr = np.array([(r[3], r[4], r[5]) for r in records], dtype=np.float32)
    meta = np.array([f"{r[0]}|{r[1]}|{r[2]}" for r in records])
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = cfg.path(stage, "transplant")
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id), cells=meta, sweep=arr,
             coeffs=np.array(coeffs), baseline_harmless_refusal=np.array(baseline),
             sources=np.array([f"{a}|{b}|{c}" for a, b, c in srcs]),
             positive_control_ok=np.array(ctrl_ok or is_first),
             lineage=np.array(cfg.lineage))
    logger.info("[%s] saved %s", stage, path)

    for (src, layer, _pos) in srcs:
        for kind in ("direction", "random"):
            sel = [r for r in records if r[0] == src and r[2] == kind]
            best = max(sel, key=lambda r: r[4])
            crossing = [r[3] for r in sel if r[4] >= cfg.induce_threshold]
            rec.result(target=stage, source=src, layer=layer, kind=kind,
                       induces=bool(crossing),
                       first_coeff=(min(crossing) if crossing else None),
                       max_induced=round(best[4], 4), coeff_at_max=best[3],
                       kl_at_max=round(best[5], 3),
                       dir_norm=round(float(np.linalg.norm(srcs[(src, layer, _pos)])), 2),
                       baseline_harmless=round(baseline, 4),
                       positive_control_ok=bool(ctrl_ok or is_first))
    del model
    torch.cuda.empty_cache()


def _norm_matched(d: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r = torch.randn(d.shape, generator=gen, dtype=torch.float32)
    return (r / (r.norm() + 1e-8) * d.norm().cpu()).to(d.device).to(d.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES "
                         "(zephyr | olmo2 | tulu2)")
    ap.add_argument("--stage", required=True, help="target model (base/sft/dpo) or 'all'")
    ap.add_argument("--unit-norm", action="store_true",
                    help="unit-normalise source directions so the coefficient IS the injected "
                         "norm, making the sweep comparable across sources (see O-47)")
    args = ap.parse_args()
    cfg = config_for(args.lineage)
    cfg.require_verified()          # conceptual blocker first ...
    logger.info("data: %s", assert_available())   # ... then the cheap file check
    sources = source_layers(cfg)
    srcs, raw_norms = load_source_directions(cfg, sources, unit_norm=args.unit_norm)
    coeffs = coeff_grid(raw_norms, args.unit_norm)
    logger.info("direction scaling: %s", "UNIT-NORM (coeff = injected norm)" if args.unit_norm
                else "RAW (Arditi default; coeff not comparable across sources)")
    logger.info("source directions: %s",
                [f"{s}@L{l}/p{p} raw_norm={raw_norms[(s, l, p)]:.1f}" for (s, l, p) in srcs])
    if args.unit_norm:
        logger.info("coefficient grid = %s x max raw norm %.1f -> %s",
                    NORM_MULTIPLES, max(raw_norms.values()),
                    [round(c, 1) for c in coeffs])
    else:
        logger.info("coefficient grid = %s (raw)", coeffs)
    if not args.unit_norm:
        logger.info("NOTE: norms differ across checkpoints, so a given coefficient is NOT "
                    "comparable across sources — read the sweep, not a single coeff. "
                    "Re-run with --unit-norm for a matched-injection comparison.")
    ckpts = dict(cfg.checkpoints)
    stages = list(ckpts) if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in ckpts:
            raise SystemExit(f"unknown stage '{s}'. known: {list(ckpts)}")
    with RunRecord(EXPERIMENT, "transplant.py", cfg, question=QUESTION + (
                       "  [unit-norm]" if args.unit_norm else "  [raw norms]"),
                   notes="The decisive cell is target=base, source=dpo: if DPO's refusal "
                         "direction induces refusal in base at acceptable KL, base already "
                         "had the machinery. Sweeping coefficients also retires the 'you only "
                         "tried coeff=1' objection to E02.") as rec:
        for s in stages:
            run_one(s, ckpts[s], cfg, srcs, rec, coeffs)


if __name__ == "__main__":
    main()
