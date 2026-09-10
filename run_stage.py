"""Compute the per-layer refusal-strength curve for ONE checkpoint, save to results/.

    python run_stage.py --stage base      # or sft / dpo (names from config.checkpoints)
    python run_stage.py --stage all       # run every checkpoint sequentially

Output: results/{stage}_refusal.npz  (bypass curve, l_star, baseline, excluded_layers)
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
from refusal_direction import eoi_len, get_mean_diff, refusal_strength_curve

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


def run_one(stage: str, model_id: str, cfg) -> None:
    set_seed(cfg.seed)
    model, tok = load_model(model_id, cfg.dtype)
    refusal_toks = [tok.encode(cfg.refusal_onset_str, add_special_tokens=False)[-1]]
    n_eoi = eoi_len(tok, cfg.template)

    harmful_tr = load_instructions("harmful_train")[: cfg.n_train]
    harmless_tr = load_instructions("harmless_train")[: cfg.n_train]
    harmful_val = load_instructions("harmful_val")[: cfg.n_val]

    logger.info("[%s] extracting directions (%d eoi positions, refusal_tok=%s)",
                stage, n_eoi, refusal_toks)
    directions = get_mean_diff(model, tok, harmful_tr, harmless_tr, cfg.template, n_eoi, cfg.batch_size)
    res = refusal_strength_curve(model, tok, directions, harmful_val, cfg.template,
                                 refusal_toks, cfg.prune_layer_pct, cfg.batch_size)

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = f"{cfg.results_dir}/{stage}_refusal.npz"
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id),
             bypass=res["bypass"], l_star=np.array(res["l_star"]),
             baseline_refusal=np.array(res["baseline_refusal"]),
             excluded_layers=res["excluded_layers"])
    logger.info("[%s] saved %s | l*=%d baseline_refusal=%.3f peak_strength=%.3f",
                stage, path, res["l_star"], res["baseline_refusal"], float(np.nanmax(res["bypass"])))
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="stage name (base/sft/dpo) or 'all'")
    args = ap.parse_args()
    cfg = DEFAULT
    ckpts = dict(cfg.checkpoints)
    stages = list(ckpts) if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in ckpts:
            raise SystemExit(f"unknown stage '{s}'. known: {list(ckpts)}")
    for s in stages:
        run_one(s, ckpts[s], cfg)


if __name__ == "__main__":
    main()
