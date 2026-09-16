"""P1-E1 — Is the harmful/harmless distinction READABLE in the base model, and is it the
SAME axis the aligned model refuses along?

    python probe_representation.py --stage all

Question: the pilot (E02) found base has no *actionable* refusal direction. This asks the
prior question — is the distinction linearly present but causally inert (the hypothesis),
or genuinely absent (which would kill it)?

Measures, per (position, layer), for every checkpoint:
  1. mass-mean probe accuracy   — same estimator as the refusal direction
  2. logistic probe accuracy    — a strictly stronger learner
  3. the mean-diff direction     — for cross-stage cosines
  4. k shuffled-label directions — the null band those cosines must clear
plus a token-length-only baseline, and layer 0 (the embedding) as the surface-feature
control built into the layer profile.

Splits, kept clean:
  fit  : harmful_train[:n_train] / harmless_train[:n_train]  — the SAME data the refusal
         directions were fit on, so a cross-stage cosine isolates the model as the only
         variable.
  test : harmful_train[n_train:] (held out, also used by the behavioural axis) and a
         count-matched slice of harmless_train. Balanced, so chance is 0.5.

Output: results/{stage}_probe.npz   Ledger: results/RUNLOG.md (see runlog.py)
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

from config import config_for
from data import assert_available, load_instructions
from probes import (cache_activations, length_baseline, logistic_accuracy,
                    mass_mean_accuracy, mass_mean_direction, null_directions)
from run_stage import load_model, set_seed
from runlog import RunRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("P1-E1")

EXPERIMENT = "P1-E1"
QUESTION = ("Is harmful-vs-harmless linearly readable in base, and is it the same axis "
            "the aligned model refuses along?")


def _splits(cfg):
    """(fit_pos, fit_neg, test_pos, test_neg). Test is held out from direction fitting."""
    harmful, harmless = load_instructions("harmful_train"), load_instructions("harmless_train")
    fit_pos, fit_neg = harmful[: cfg.n_train], harmless[: cfg.n_train]
    test_pos = harmful[cfg.n_train:]
    test_neg = harmless[cfg.n_train: cfg.n_train + len(test_pos)]   # count-matched -> chance 0.5
    if len(test_neg) < len(test_pos):
        test_pos = test_pos[: len(test_neg)]
    return fit_pos, fit_neg, test_pos, test_neg


def run_one(stage: str, model_id: str, cfg, rec: RunRecord) -> None:
    set_seed(cfg.seed)
    model, tok = load_model(model_id, cfg.dtype)
    template, _want_id, n_eoi_stage, is_ov = cfg.regime(stage)
    if is_ov:
        logger.warning("[%s] REGIME OVERRIDE: template=%r n_eoi=%d — activations come from a "
                       "different prompt format than the other stages (O-56/O-57), so "
                       "cross-stage cosines against this stage are NOT format-matched.",
                       stage, template, n_eoi_stage)
    fit_pos, fit_neg, test_pos, test_neg = _splits(cfg)
    logger.info("[%s] fit %d/%d | test %d/%d (chance=%.3f)", stage, len(fit_pos), len(fit_neg),
                len(test_pos), len(test_neg), len(test_pos) / (len(test_pos) + len(test_neg)))

    A = {}
    for name, prompts in (("fit_pos", fit_pos), ("fit_neg", fit_neg),
                          ("test_pos", test_pos), ("test_neg", test_neg)):
        logger.info("[%s] caching activations: %s (n=%d)", stage, name, len(prompts))
        A[name] = cache_activations(model, tok, prompts, template, n_eoi_stage,
                                    cfg.batch_size).numpy()
    n_pos, n_layers = A["fit_pos"].shape[1], A["fit_pos"].shape[2]

    acc_mm = np.full((n_pos, n_layers), np.nan)
    acc_lr = np.full((n_pos, n_layers), np.nan)
    dirs = np.zeros((n_pos, n_layers, A["fit_pos"].shape[3]), dtype=np.float32)
    nulls = np.zeros((cfg.n_null, n_pos, n_layers, A["fit_pos"].shape[3]), dtype=np.float16)
    rng = np.random.default_rng(cfg.seed)

    for p in range(n_pos):
        for l in range(n_layers):
            tp, tn = A["fit_pos"][:, p, l, :], A["fit_neg"][:, p, l, :]
            ep, en = A["test_pos"][:, p, l, :], A["test_neg"][:, p, l, :]
            acc_mm[p, l] = mass_mean_accuracy(tp, tn, ep, en)
            acc_lr[p, l] = logistic_accuracy(tp, tn, ep, en, seed=cfg.seed)
            dirs[p, l] = mass_mean_direction(tp, tn)
            pooled = np.concatenate([tp, tn])
            nulls[:, p, l, :] = null_directions(pooled, len(tp), cfg.n_null, rng).astype(np.float16)
        logger.info("[%s] pos %d/%d done | best mass-mean %.3f | best logistic %.3f",
                    stage, p + 1, n_pos, np.nanmax(acc_mm[p]), np.nanmax(acc_lr[p]))

    # token-length-only floor, and the layer-0 (embedding) surface-feature control
    def toklen(xs):
        return np.array([len(tok.encode(template.format(instruction=i))) for i in xs])
    len_acc = length_baseline(toklen(fit_pos), toklen(fit_neg), toklen(test_pos), toklen(test_neg))

    best_mm = np.nanmax(acc_mm, axis=0)
    best_lr = np.nanmax(acc_lr, axis=0)
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = cfg.path(stage, "probe")
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id),
             acc_mass_mean=acc_mm, acc_logistic=acc_lr, directions=dirs, null_directions=nulls,
             length_baseline=np.array(len_acc), n_fit=np.array(len(fit_pos)),
             n_test=np.array(len(test_pos)))
    logger.info("[%s] saved %s | mass-mean peak=%.3f @L%d | logistic peak=%.3f @L%d | "
                "L0=%.3f | length-only=%.3f", stage, path, best_mm.max(), int(best_mm.argmax()),
                best_lr.max(), int(best_lr.argmax()), best_lr[0], len_acc)

    rec.result(stage=stage, mass_mean_peak=round(float(best_mm.max()), 4),
               mass_mean_peak_layer=int(best_mm.argmax()),
               logistic_peak=round(float(best_lr.max()), 4),
               logistic_peak_layer=int(best_lr.argmax()),
               logistic_L0=round(float(best_lr[0]), 4),
               length_only_baseline=round(float(len_acc), 4),
               n_fit=len(fit_pos), n_test_per_class=len(test_pos))
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES "
                         "(zephyr | olmo2 | tulu2)")
    ap.add_argument("--stage", required=True, help="stage name (base/sft/dpo) or 'all'")
    args = ap.parse_args()
    # Cheap checks first: a missing clone costs nothing to detect and a full weight
    # download to discover late (hit on a pod, 2026-09-13).
    cfg = config_for(args.lineage)
    cfg.require_verified()          # conceptual blocker first ...
    logger.info("data: %s", assert_available())   # ... then the cheap file check
    ckpts = dict(cfg.checkpoints)
    stages = list(ckpts) if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in ckpts:
            raise SystemExit(f"unknown stage '{s}'. known: {list(ckpts)}")
    with RunRecord(EXPERIMENT, "probe_representation.py", cfg, question=QUESTION,
                   notes="Accuracy alone is near-certain to be high and proves little; the "
                         "informative outputs are the layer profile (L0 vs peak) and the "
                         "cross-stage cosines computed by aggregate_probe.py.") as rec:
        for s in stages:
            run_one(s, ckpts[s], cfg, rec)


if __name__ == "__main__":
    main()
