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

from config import DEFAULT, config_for
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

COEFFS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


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
            pos_star = cfg.n_eoi - 1          # last eoi position; no validated pos*
            logger.warning("[%s] no filtered l* — falling back to the unfiltered argmax L%d. "
                           "This is NOT a validated refusal layer.", stage, l_star)
        out.append((stage, l_star, pos_star))
    return out


def load_source_directions(cfg, sources, unit_norm: bool = False) -> dict[tuple[str, int], np.ndarray]:
    """Reuse the P1-E1 probe directions: identical estimator, data and hook point as E02's
    refusal directions (mean-diff harmful-harmless at resid_pre over eoi positions)."""
    out = {}
    for stage, layer, pos_idx in sources:
        path = cfg.path(stage, "probe")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — run probe_representation.py --stage all first")
        d = np.load(path, allow_pickle=True)["directions"]
        v = d[pos_idx, layer].astype(np.float32)
        if unit_norm:
            # Norms differ 1.1 / 7.4 / 4.4 across base/sft/dpo, so a raw coefficient is not
            # comparable across sources. Unit-normalising makes the coefficient the INJECTED
            # NORM, which is. Changes what the numbers mean -- report which mode was used.
            v = v / (np.linalg.norm(v) + 1e-8)
        out[(stage, layer)] = v
    return out


def sweep_cell(model, tok, direction: torch.Tensor, layer: int, harmless, template,
               refusal_toks, base_harmless_logits, batch_size):
    """Induced refusal and KL on harmless prompts, across COEFFS, for ONE direction."""
    rows = []
    for c in COEFFS:
        h = _addition_handles(model, direction, coeff=c, layer=layer)
        try:
            lg = _last_logits(model, tok, harmless, template, batch_size)
            rows.append((c, _mean_refusal(lg, refusal_toks), kl_last(base_harmless_logits, lg)))
        finally:
            for x in h:
                x.remove()
    return rows


def run_one(stage: str, model_id: str, cfg, srcs, rec: RunRecord) -> None:
    set_seed(cfg.seed)
    model, tok = load_model(model_id, cfg.dtype)
    refusal_toks = [resolve_refusal_token(tok, cfg.refusal_token_piece, cfg.expected_refusal_id)]
    harmless = load_instructions("harmless_val")[: cfg.n_transplant]

    base_lg = _last_logits(model, tok, harmless, cfg.template, cfg.batch_size)
    baseline = _mean_refusal(base_lg, refusal_toks)
    logger.info("[%s] baseline refusal on HARMLESS = %.3f over n=%d (induction must beat %.2f)",
                stage, baseline, len(harmless), cfg.induce_threshold)

    gen = torch.Generator().manual_seed(cfg.seed)
    records = []
    for (src, layer), vec in srcs.items():
        d = torch.from_numpy(vec).to(model.device)
        for kind, direction in (("direction", d),
                                ("random", _norm_matched(d, gen))):
            for c, ref, kl in sweep_cell(model, tok, direction, layer, harmless, cfg.template,
                                         refusal_toks, base_lg, cfg.batch_size):
                records.append((src, layer, kind, c, ref, kl))
            last = records[-len(COEFFS):]      # exactly this cell's coefficient sweep
            best = max(last, key=lambda r: r[4])
            ok = [c for _, _, _, c, r, _ in last if r >= cfg.induce_threshold]
            logger.info("[%s] src=%-4s L%-2d %-9s | max induced %+.3f @coeff %.1f (KL %.2f) | %s",
                        stage, src, layer, kind, best[4], best[3], best[5],
                        f"INDUCES (first at coeff {min(ok)})" if ok
                        else "never crosses threshold")

    arr = np.array([(r[3], r[4], r[5]) for r in records], dtype=np.float32)
    meta = np.array([f"{r[0]}|{r[1]}|{r[2]}" for r in records])
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = cfg.path(stage, "transplant")
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id), cells=meta, sweep=arr,
             coeffs=np.array(COEFFS), baseline_harmless_refusal=np.array(baseline),
             sources=np.array([f"{a}|{b}|{c}" for a, b, c in sources]),
             lineage=np.array(cfg.lineage))
    logger.info("[%s] saved %s", stage, path)

    for (src, layer) in srcs:
        for kind in ("direction", "random"):
            sel = [r for r in records if r[0] == src and r[2] == kind]
            best = max(sel, key=lambda r: r[4])
            crossing = [r[3] for r in sel if r[4] >= cfg.induce_threshold]
            rec.result(target=stage, source=src, layer=layer, kind=kind,
                       induces=bool(crossing),
                       first_coeff=(min(crossing) if crossing else None),
                       max_induced=round(best[4], 4), coeff_at_max=best[3],
                       kl_at_max=round(best[5], 3),
                       dir_norm=round(float(np.linalg.norm(srcs[(src, layer)])), 2),
                       baseline_harmless=round(baseline, 4))
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
    srcs = load_source_directions(cfg, sources, unit_norm=args.unit_norm)
    logger.info("direction scaling: %s", "UNIT-NORM (coeff = injected norm)" if args.unit_norm
                else "RAW (Arditi default; coeff not comparable across sources)")
    logger.info("source directions: %s",
                [f"{s}@L{l} norm={np.linalg.norm(v):.1f}" for (s, l), v in srcs.items()])
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
            run_one(s, ckpts[s], cfg, srcs, rec)


if __name__ == "__main__":
    main()
