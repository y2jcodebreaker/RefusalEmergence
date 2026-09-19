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
import json
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


def coeff_grid(raw_norms: dict, unit_norm: bool, own_norms: bool = True) -> tuple[float, ...]:
    """Coefficients to sweep. Raw mode: Arditi's fixed grid.

    Unit-norm mode: multiples of the LARGEST source norm, which keeps injected norm matched
    across sources, PLUS each source's own raw norm.

    Why the second part. Anchoring only on the maximum means a small-norm source is swept at
    wild multiples of its own scale. Zephyr's norms are 1.1 / 4.4 / 7.4, so the top of the grid
    is 54x base's own norm but only 8x SFT's -- a 6.7x disparity in how hard each direction is
    pushed relative to where it naturally lives (OLMo 2's disparity was 2.25x, which is why this
    only showed up on Zephyr). For the NEGATIVE claim about base that is a strength: we pushed
    far past natural and nothing happened. For COMPARING sources it is a confound, and it
    plausibly explains Zephyr's much larger nulls (+-1.4 to 2.8 against OLMo 2's +-0.4 to 1.2):
    at 54x you are battering the model, so arbitrary directions do a lot too.

    Adding every source's own norm to the SHARED grid gives each direction a 1x-its-own-norm
    point -- the natural operating point, and the one to quote -- without giving up matched
    injection, since all sources are still swept at all the same coefficients."""
    if not unit_norm:
        return COEFFS
    anchor = max(raw_norms.values())
    grid = {round(m * anchor, 4) for m in NORM_MULTIPLES}
    if own_norms:
        grid |= {round(v, 4) for v in raw_norms.values()}
    return tuple(sorted(grid))


def source_layers(cfg, by: str = "ablation") -> list[tuple[str, int, int]]:
    """(stage, layer, pos_idx) per source, READ FROM each stage's saved sweep — never
    hardcoded.

    `by="ablation"` (default) uses the Arditi-filtered l*, falling back to the unfiltered
    argmax when nothing passed the filters (base) — flagged in the log, because such a layer is
    "best among allowed", not a validated refusal layer.

    `by="induce"` picks the cell that maximises the INDUCE surface instead. This exists to
    close a circularity: base has l* = -1 *because* no cell passes the induce criterion, so
    "base has no valid direction" and "base's direction does not induce" risk being the same
    statement, and the direction we transplant by default is the argmax of the ABLATION surface
    — not of the axis we score. Verified 2026-09-17: OLMo 2 base transplants (pos 1, L23) while
    its induce-optimal cell is (pos 1, L19); Zephyr L15 vs L20. Both remain far below threshold,
    so the conclusion is very likely safe — but it had not been TESTED. Reporting both turns
    "no direction passed our filters" into "base's best candidate BY THE METRIC WE SCORE still
    does nothing"."""
    if by not in ("ablation", "induce"):
        raise SystemExit(f"--source-by must be 'ablation' or 'induce', not {by!r}")
    out = []
    for stage in cfg.stages:
        path = cfg.path(stage, "refusal")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — run run_stage.py --lineage {cfg.lineage} "
                             f"--stage all first (transplant needs each stage's l*)")
        d = np.load(path, allow_pickle=True)
        l_star, pos_star = int(d["l_star"]), int(d["pos_star"])
        if by == "induce":
            steer, excl = d["steer"], d["excluded_layers"]
            keep = np.ones(steer.shape[1], dtype=bool)
            if excl.size:
                keep[excl.astype(int)] = False
            masked = np.where(keep[None, :], steer, -np.inf)
            pos_star, l_star = (int(v) for v in np.unravel_index(np.argmax(masked),
                                                                 masked.shape))
            logger.info("[%s] source by INDUCE argmax: (pos %d, L%d), steer=%+.3f "
                        "(ablation-selected would be L%d)", stage, pos_star, l_star,
                        float(steer[pos_star, l_star]),
                        int(d["l_star"]) if int(d["l_star"]) >= 0 else int(d["naive_l_star"]))
            out.append((stage, l_star, pos_star))
            continue
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


def load_null_directions(cfg, sources, k: int, unit_norm: bool = False) -> dict:
    """(stage, layer, pos) -> (k, d) SHUFFLED-LABEL mean-diff directions at the same cell.

    The harder null. An isotropic Gaussian points mostly into directions the residual stream
    barely uses -- activations are strongly anisotropic, which is the same fact that made
    cross-checkpoint cosine useless (O-49). Beating isotropic noise is therefore easy, and on
    Zephyr it showed: base's own direction, which fails every other test, scored z = +3.0
    against isotropic noise, and so did sft's (+2.5) and dpo's (+2.9) -- a uniform band that
    tracks "is a mean-diff vector" rather than "is a refusal vector" (2026-09-17).

    A direction fitted with the SAME estimator on RANDOMLY SHUFFLED labels over the SAME
    activations encodes nothing about harmfulness but inherits the data's geometry exactly.
    Already computed and stored by probe_representation.py as `null_directions`."""
    out = {}
    for stage, layer, pos_idx in sources:
        path = cfg.path(stage, "probe")
        n = np.load(path, allow_pickle=True)["null_directions"]      # (k_stored, pos, layer, d)
        if k > n.shape[0]:
            raise SystemExit(
                f"[{stage}] asked for {k} shuffled-label nulls but only {n.shape[0]} are "
                f"stored in {path}. Raise Config.n_null and re-run probe_representation.py.")
        v = n[:k, pos_idx, layer].astype(np.float32)
        if unit_norm:
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        out[(stage, layer, pos_idx)] = v
    return out


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


def cell_effect(records, src: str, baseline: float,
                null_prefix: str = "random") -> dict:
    """Effect size of one transplant cell against its OWN null distribution.

    The absolute crossing score ("does it reach 0?") is confounded by where the target's
    baseline sits: OLMo 2's DPO and RLVR start 3 points further from refusing than its SFT
    does, so the same absolute peak is a much larger causal effect. Δ from the target's own
    baseline is the effect size; the k random arms give it a scale.

    Returns delta (real), null_mean, null_sd, z = (delta - mean)/sd, n_draws, and how many
    random draws themselves crossed. With one draw sd is undefined and z is None -- which is
    the state that left SP4 unresolved (nulls scattered +0.22 / +2.83 / +0.74)."""
    def best(kind_test):
        vals = [r[4] for r in records if r[0] == src and kind_test(r[2])]
        return max(vals) if vals else float("nan")

    delta = best(lambda k: k == "direction") - baseline
    draws = sorted({r[2] for r in records
                    if r[0] == src and r[2].startswith(null_prefix)})
    nulls = [best(lambda k, d=d: k == d) - baseline for d in draws]
    if not nulls:
        return {"null": null_prefix, "delta": delta, "null_mean": None,
                "null_sd": None, "z": None, "n_draws": 0, "n_null_crossing": 0}
    mu = sum(nulls) / len(nulls)
    sd = (sum((v - mu) ** 2 for v in nulls) / (len(nulls) - 1)) ** 0.5 if len(nulls) > 1 else None
    return {"null": null_prefix, "delta": delta, "null_mean": mu, "null_sd": sd,
            "z": ((delta - mu) / sd) if (sd and sd > 0) else None,
            "n_draws": len(nulls),
            "n_null_crossing": sum(1 for v in nulls if v + baseline >= 0.0)}


def positive_control(records, stage: str, threshold: float):
    """(ok, best_induced, self_cells) — did the target's OWN direction induce refusal in it?

    `records` rows are (src, layer, kind, coeff, induced_refusal, kl). The self-cell is the
    positive control for the whole matrix: see the comment at its call site."""
    self_cells = [r for r in records if r[0] == stage and r[2] == "direction"]
    best = max((r[4] for r in self_cells), default=float("nan"))
    return bool(self_cells) and best >= threshold, best, self_cells


def run_one(stage: str, model_id: str, cfg, srcs, rec: RunRecord, coeffs=COEFFS,
            n_null: int = 1, nulls: dict | None = None,
            null_kinds: tuple[str, ...] = ("random",)) -> None:
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
        # ONE real arm and n_null independent random arms. A single random draw cannot scale
        # an effect: the three self-cells' nulls came back +0.22 / +2.83 / +0.74, a spread
        # comparable to the differences being compared, which is why SP4 was unresolvable
        # (O-52's lesson: size n against the effect size). Arms are labelled random0..randomk-1
        # so every consumer that filters on "direction" is unaffected.
        arms = [("direction", d)]
        if "random" in null_kinds:
            arms += [(f"random{j}", _norm_matched(d, gen)) for j in range(n_null)]
        if "shuffled" in null_kinds:
            # Norm-matched to the REAL direction so the only thing that differs between arms
            # is orientation -- in unit-norm mode both are already unit length.
            for j, nv in enumerate(nulls[(src, layer, _pos)]):
                t = torch.from_numpy(nv).to(model.device)
                t = t / (t.norm() + 1e-8) * d.norm()
                arms.append((f"shuffled{j}", t.to(d.dtype)))
        for kind, direction in arms:
            for c, ref, kl in sweep_cell(model, tok, direction, layer, harmless, template,
                                         refusal_toks, base_lg, cfg.batch_size, coeffs):
                records.append((src, layer, kind, c, ref, kl))
            last = records[-len(coeffs):]      # exactly this cell's coefficient sweep
            best = max(last, key=lambda r: r[4])
            ok = [c for _, _, _, c, r, _ in last if r >= cfg.induce_threshold]
            if kind == "direction":
                logger.info("[%s] src=%-4s L%-2d %-9s | max induced %+.3f @coeff %.1f "
                            "(KL %.2f) | %s", stage, src, layer, kind, best[4], best[3],
                            best[5], f"INDUCES (first at coeff {min(ok)})" if ok
                            else "never crosses threshold")
        # One summary line per null FAMILY instead of k noisy ones per arm.
        for nk in null_kinds:
            eff = cell_effect(records, src, baseline, null_prefix=nk)
            if not eff["n_draws"]:
                continue
            logger.info("[%s] src=%-4s L%-2d effect %-8s | delta %+.2f vs null %+.2f+-%s over "
                        "%d draws%s | %d/%d nulls crossed", stage, src, layer, nk,
                        eff["delta"], eff["null_mean"],
                        f"{eff['null_sd']:.2f}" if eff["null_sd"] is not None else "n/a",
                        eff["n_draws"],
                        f" | z={eff['z']:+.1f}" if eff["z"] is not None else "",
                        eff["n_null_crossing"], eff["n_draws"])

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
             n_null_draws=np.array(n_null),
             effects=np.array(json.dumps(
                 {f"{s_}|{nk}": cell_effect(records, s_, baseline, null_prefix=nk)
                  for s_ in {r[0] for r in records} for nk in null_kinds})),
             lineage=np.array(cfg.lineage))
    logger.info("[%s] saved %s", stage, path)

    for (src, layer, _pos) in srcs:
        effs = {nk: cell_effect(records, src, baseline, null_prefix=nk) for nk in null_kinds}
        eff = effs[null_kinds[0]]
        kinds = ["direction"] + sorted({r[2] for r in records if r[0] == src
                                        and r[2] != "direction"})
        for kind in kinds:
            sel = [r for r in records if r[0] == src and r[2] == kind]
            best = max(sel, key=lambda r: r[4])
            crossing = [r[3] for r in sel if r[4] >= cfg.induce_threshold]
            row = dict(target=stage, source=src, layer=layer, kind=kind,
                       induces=bool(crossing),
                       first_coeff=(min(crossing) if crossing else None),
                       max_induced=round(best[4], 4), coeff_at_max=best[3],
                       kl_at_max=round(best[5], 3),
                       dir_norm=round(float(np.linalg.norm(srcs[(src, layer, _pos)])), 2),
                       baseline_harmless=round(baseline, 4),
                       positive_control_ok=bool(ctrl_ok or is_first))
            if kind == "direction":      # the effect size belongs to the CELL, not an arm
                row["delta"] = round(eff["delta"], 4)
                for nk, e in effs.items():
                    row.update({
                        f"{nk}_mean": (round(e["null_mean"], 4)
                                       if e["null_mean"] is not None else None),
                        f"{nk}_sd": round(e["null_sd"], 4) if e["null_sd"] is not None else None,
                        f"{nk}_z": round(e["z"], 3) if e["z"] is not None else None,
                        f"{nk}_draws": e["n_draws"],
                        f"{nk}_crossing": e["n_null_crossing"]})
            rec.result(**row)
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
    ap.add_argument("--source-by", default="ablation", choices=("ablation", "induce"),
                    help="which cell to take each source direction from. 'ablation' (default) "
                         "is the Arditi-filtered l*. 'induce' is the argmax of the induce "
                         "surface -- run it to close the circularity in C2 (see source_layers).")
    ap.add_argument("--no-own-norms", action="store_true",
                    help="omit each source's own raw norm from the coefficient grid. On by "
                         "default: it gives every direction a 1x-its-own-norm point, which is "
                         "the natural operating point and the one worth quoting.")
    ap.add_argument("--null", default="both", choices=("random", "shuffled", "both"),
                    help="null family. 'random' = isotropic norm-matched (Arditi convention, "
                         "and what earlier runs used). 'shuffled' = same estimator fitted on "
                         "SHUFFLED LABELS, which inherits the data's anisotropic geometry and "
                         "is the harder control. 'both' (default) reports each separately.")
    ap.add_argument("--n-control", type=int, default=None,
                    help="independent norm-matched random draws per cell (default: "
                         "cfg.n_control). ONE draw cannot scale an effect -- see P1-E2c")
    ap.add_argument("--unit-norm", action="store_true",
                    help="unit-normalise source directions so the coefficient IS the injected "
                         "norm, making the sweep comparable across sources (see O-47)")
    args = ap.parse_args()
    cfg = config_for(args.lineage)
    cfg.require_verified()          # conceptual blocker first ...
    logger.info("data: %s", assert_available())   # ... then the cheap file check
    logger.info("source selection: %s", args.source_by.upper())
    n_null = args.n_control if args.n_control is not None else cfg.n_control
    null_kinds = ("random", "shuffled") if args.null == "both" else (args.null,)
    if n_null < 2:
        logger.warning("--n-control=%d: with fewer than 2 draws the null has no spread, so "
                       "effect sizes CANNOT be compared across stages (this is what left SP4 "
                       "unresolved). Crossing verdicts are still valid.", n_null)
    logger.info("null draws per cell: %d", n_null)
    sources = source_layers(cfg, by=args.source_by)
    srcs, raw_norms = load_source_directions(cfg, sources, unit_norm=args.unit_norm)
    nulls = (load_null_directions(cfg, sources, n_null, unit_norm=args.unit_norm)
             if "shuffled" in null_kinds else None)
    logger.info("null families: %s x %d draws%s", " + ".join(null_kinds), n_null,
                "  (shuffled = same estimator on shuffled labels; shares the data's "
                "anisotropic geometry, so it is the harder control)"
                if "shuffled" in null_kinds else "")
    coeffs = coeff_grid(raw_norms, args.unit_norm, own_norms=not args.no_own_norms)
    logger.info("direction scaling: %s", "UNIT-NORM (coeff = injected norm)" if args.unit_norm
                else "RAW (Arditi default; coeff not comparable across sources)")
    logger.info("source directions: %s",
                [f"{s}@L{l}/p{p} raw_norm={raw_norms[(s, l, p)]:.1f}" for (s, l, p) in srcs])
    if args.unit_norm:
        own = sorted({round(v, 1) for v in raw_norms.values()})
        logger.info("coefficient grid = %s x max raw norm %.1f, plus each source's own norm "
                    "%s -> %s", NORM_MULTIPLES, max(raw_norms.values()), own,
                    [round(c, 1) for c in coeffs])
        for (s_, l_, p_), v in raw_norms.items():
            logger.info("  %-4s@L%-2d raw_norm=%5.1f -> grid spans %.1fx to %.1fx ITS OWN norm",
                        s_, l_, v, min(coeffs) / v, max(coeffs) / v)
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
            run_one(s, ckpts[s], cfg, srcs, rec, coeffs, n_null=n_null,
                    nulls=nulls, null_kinds=null_kinds)


if __name__ == "__main__":
    main()
