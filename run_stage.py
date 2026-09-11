"""Compute the per-layer refusal-strength curve for ONE checkpoint, save to results/.

    python run_stage.py --stage base            # or sft / dpo (names from config.checkpoints)
    python run_stage.py --stage all             # run every checkpoint sequentially
    python run_stage.py --stage all --control   # + norm-matched random-direction control

Output: results/{stage}_refusal.npz  (bypass curve, l_star, baseline, excluded_layers,
and with --control: control_bypass, control_curves)
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch

from config import DEFAULT
from data import load_instructions
from refusal_direction import (eoi_len, get_mean_diff, norm_matched_random,
                               refusal_strength_curve, resolve_refusal_token)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_stage")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def load_model(model_id: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    logger.info("loading %s (%s)", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto").eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def run_one(stage: str, model_id: str, cfg, control: bool = False) -> None:
    set_seed(cfg.seed)
    model, tok = load_model(model_id, cfg.dtype)
    refusal_toks = [resolve_refusal_token(tok, cfg.refusal_token_piece, cfg.expected_refusal_id)]
    # PINNED (not tokenizer-derived): eoi_len differs 9 (base) vs 10 (SFT/DPO) because the
    # Zephyr tokenizers insert a phantom '' token. A stage-varying window would invalidate
    # the cross-stage comparison. See config.N_EOI_FIXED.
    n_eoi = cfg.n_eoi
    logger.info("[%s] refusal_tok=%s (%r) | pinned n_eoi=%d | tokenizer-derived would be %d",
                stage, refusal_toks, tok.decode(refusal_toks), n_eoi, eoi_len(tok, cfg.template))

    harmful_tr = load_instructions("harmful_train")[: cfg.n_train]
    harmless_tr = load_instructions("harmless_train")[: cfg.n_train]
    harmful_val = load_instructions("harmful_val")[: cfg.n_val]

    logger.info("[%s] extracting directions (%d eoi positions)", stage, n_eoi)
    directions = get_mean_diff(model, tok, harmful_tr, harmless_tr, cfg.template, n_eoi, cfg.batch_size)
    res = refusal_strength_curve(model, tok, directions, harmful_val, cfg.template,
                                 refusal_toks, cfg.prune_layer_pct, cfg.batch_size)

    extra = {}
    if control:
        # IDENTICAL sweep, only the directions differ -> any gap is about orientation.
        gen = torch.Generator().manual_seed(cfg.seed)
        curves = []
        for k in range(cfg.n_control):
            logger.info("[%s] control sweep %d/%d (norm-matched random)", stage, k + 1, cfg.n_control)
            rand_dirs = norm_matched_random(directions, gen)
            curves.append(refusal_strength_curve(model, tok, rand_dirs, harmful_val, cfg.template,
                                                 refusal_toks, cfg.prune_layer_pct,
                                                 cfg.batch_size)["bypass"])
        extra["control_bypass"] = np.mean(curves, axis=0)
        extra["control_curves"] = np.asarray(curves)
        logger.info("[%s] control peak=%.3f  vs  refusal peak=%.3f",
                    stage, float(np.nanmax(extra["control_bypass"])),
                    float(np.nanmax(res["bypass"])))

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = f"{cfg.results_dir}/{stage}_refusal.npz"
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id),
             bypass=res["bypass"], l_star=np.array(res["l_star"]),
             baseline_refusal=np.array(res["baseline_refusal"]),
             excluded_layers=res["excluded_layers"], **extra)
    logger.info("[%s] saved %s | l*=%d baseline_refusal=%.3f peak_strength=%.3f",
                stage, path, res["l_star"], res["baseline_refusal"], float(np.nanmax(res["bypass"])))
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="stage name (base/sft/dpo) or 'all'")
    ap.add_argument("--control", action="store_true",
                    help="also run the norm-matched random-direction negative control")
    args = ap.parse_args()
    cfg = DEFAULT
    ckpts = dict(cfg.checkpoints)
    stages = list(ckpts) if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in ckpts:
            raise SystemExit(f"unknown stage '{s}'. known: {list(ckpts)}")
    for s in stages:
        run_one(s, ckpts[s], cfg, control=args.control)


if __name__ == "__main__":
    main()
