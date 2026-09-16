"""Inspect the Arditi selection surfaces (KL, induce) saved by run_stage.py. No GPU.

    python show_filters.py                 # all stages
    python show_filters.py --stage base

Answers the only question that matters when l* = -1: WHICH criterion failed, and by how
much. "No direction survives the KL bound" and "no direction induces refusal" are entirely
different claims about the model — the first is a threshold-transfer problem, the second is
a statement that the model has no refusal mode to induce.

Also sweeps kl_threshold to show what WOULD pass at each value. That is diagnostic only:
loosening the bound to manufacture a result is exactly the move this whole filter exists to
prevent. Read it to understand the failure, not to tune your way out of it.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from config import config_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES "
                         "(zephyr | olmo2 | tulu2)")
    ap.add_argument("--stage", default=None)
    args = ap.parse_args()

    cfg = config_for(args.lineage)
    paths = sorted(glob.glob(f"{cfg.results_dir}/{cfg.lineage}_*_refusal.npz"))
    if not paths:
        raise SystemExit("no results — run run_stage.py first")

    for p in paths:
        d = np.load(p, allow_pickle=True)
        stage = str(d["stage"])
        if args.stage and stage != args.stage:
            continue
        if "kl" not in d:
            print(f"\n[{stage}] no filter data (produced before the Arditi filters landed)")
            continue

        kl, steer = d["kl"], d["steer"]
        l_star = int(d["l_star"])
        excl = d["excluded_layers"]
        unpruned = np.ones_like(kl, dtype=bool)
        if excl.size:
            unpruned[:, excl.astype(int)] = False

        print(f"\n{'=' * 78}\n[{stage}]  l*={l_star}"
              f"{'  (NO VALID DIRECTION)' if l_star < 0 else ''}"
              f"   naive argmax={int(d['naive_l_star'])}\n{'=' * 78}")
        print(f"  unpruned cells: {int(unpruned.sum())} of {kl.size}")
        print(f"  KL     min={np.nanmin(kl[unpruned]):.4f}  median={np.nanmedian(kl[unpruned]):.4f}"
              f"  max={np.nanmax(kl[unpruned]):.4f}   (need <= {cfg.kl_threshold})")
        print(f"  steer  min={np.nanmin(steer[unpruned]):.4f}  median={np.nanmedian(steer[unpruned]):.4f}"
              f"  max={np.nanmax(steer[unpruned]):.4f}   (need >= {cfg.induce_threshold})")

        kl_ok = (kl <= cfg.kl_threshold) & unpruned
        ind_ok = (steer >= cfg.induce_threshold) & unpruned
        print(f"\n  pass KL only   : {int(kl_ok.sum()):4d}")
        print(f"  pass induce only: {int(ind_ok.sum()):4d}")
        print(f"  pass BOTH      : {int((kl_ok & ind_ok).sum()):4d}")

        # What would pass at other KL bounds — diagnostic, NOT a tuning knob.
        print("\n  cells passing BOTH as the KL bound is relaxed (diagnostic only):")
        for t in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
            n = int((((kl <= t) & unpruned) & ind_ok).sum())
            mark = "  <- current" if abs(t - cfg.kl_threshold) < 1e-9 else ""
            print(f"    KL <= {t:<5}: {n:4d}{mark}")
        if not ind_ok.any():
            print("\n  NOTE: induce fails EVERYWHERE. No KL bound can rescue this — adding the\n"
                  "        direction never drives refusal on harmless prompts above "
                  f"{cfg.induce_threshold}.\n"
                  "        For a base model that is a plausible substantive result, not a bug:\n"
                  "        there is no refusal mode to induce yet.")

        # per-layer best steer, to see whether induce fails everywhere or just late
        print("\n  best steer per layer (max over positions):")
        keep = unpruned.any(axis=0)
        best = np.full(steer.shape[1], np.nan)
        best[keep] = np.nanmax(steer[:, keep], axis=0)
        for i in range(0, len(best), 8):
            chunk = " ".join(f"L{j}:{best[j]:+.2f}" for j in range(i, min(i + 8, len(best)))
                             if not np.isnan(best[j]))
            if chunk:
                print(f"    {chunk}")


if __name__ == "__main__":
    main()
