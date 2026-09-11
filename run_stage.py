"""Compute the per-layer refusal-strength curve for ONE checkpoint, save to results/.

    python run_stage.py --stage base            # or sft / dpo (names from config.checkpoints)
    python run_stage.py --stage all             # run every checkpoint sequentially
    python run_stage.py --stage all --control --behavioral   # + control + substring rates

Output: results/{stage}_refusal.npz  (bypass curve, l_star, baseline, excluded_layers,
and with --control: control_bypass, control_curves)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from config import DEFAULT
from data import load_instructions
from refusal_direction import (eoi_len, get_mean_diff, norm_matched_random,
                               refusal_strength_curve, resolve_refusal_token)
from refusal_substring import behavioral_rates

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


def run_one(stage: str, model_id: str, cfg, control: bool = False,
            behavioral: bool = False) -> None:
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
    harmless_val = load_instructions("harmless_val")[: cfg.n_val]   # KL side-effect check

    logger.info("[%s] extracting directions (%d eoi positions)", stage, n_eoi)
    directions = get_mean_diff(model, tok, harmful_tr, harmless_tr, cfg.template, n_eoi, cfg.batch_size)
    res = refusal_strength_curve(model, tok, directions, harmful_val, cfg.template,
                                 refusal_toks, cfg.prune_layer_pct, cfg.batch_size,
                                 harmless_val=harmless_val, kl_threshold=cfg.kl_threshold,
                                 induce_threshold=cfg.induce_threshold, filtered=True)
    valid_direction = res["l_star"] >= 0
    if not valid_direction:
        logger.warning("[%s] no valid direction — saving kl/steer/valid anyway so the failure "
                       "can be diagnosed; skipping the behavioral check.", stage)

    extra = {}
    if behavioral:
        # Second, independent axis: does the model's TEXT actually refuse, and does ablating
        # the direction stop it? Uses the best (pos, layer) direction for this checkpoint.
        # The BASELINE rate needs no direction, so it is measured even when l* = -1;
        # only the ablated half is skipped. Losing base's behavioural baseline to a failed
        # direction search would throw away a data point the search has nothing to do with.
        p = int(res["pos_star"]) if valid_direction else -1
        abl_dir = directions[p, res["l_star"]] if valid_direction else None
        # Decoupled from n_val: a rate needs n, the causal sweep does not (see config).
        # Source is the UNUSED TAIL of harmful_train (260 total, only the first n_train=128
        # fit the direction) -> 132 prompts touched by nothing: not by direction fitting,
        # not by l* selection. harmful_val is only 39 and its head drives l*, so it is both
        # too small and not fully clean for this.
        beh_prompts = load_instructions("harmful_train")[cfg.n_train:]
        if cfg.n_behavioral:
            beh_prompts = beh_prompts[: cfg.n_behavioral]
        logger.info("[%s] behavioral check @ %s on n=%d prompts", stage,
                    f"(pos={p - n_eoi}, layer={res['l_star']})" if valid_direction
                    else "BASELINE ONLY (no valid direction)", len(beh_prompts))
        b_rate, a_rate, samples = behavioral_rates(
            model, tok, beh_prompts, cfg.template, abl_dir,
            cfg.gen_max_new_tokens, cfg.batch_size, cfg.n_sample_completions)
        extra["n_behavioral"] = np.array(len(beh_prompts))
        extra["substring_baseline_rate"] = np.array(b_rate)
        extra["substring_ablated_rate"] = np.array(a_rate)
        extra["sample_completions"] = np.array(json.dumps(samples))

    if control:
        # IDENTICAL sweep, only the directions differ -> any gap is about orientation.
        gen = torch.Generator().manual_seed(cfg.seed)
        curves = []
        for k in range(cfg.n_control):
            logger.info("[%s] control sweep %d/%d (norm-matched random)", stage, k + 1, cfg.n_control)
            rand_dirs = norm_matched_random(directions, gen)
            curves.append(refusal_strength_curve(model, tok, rand_dirs, harmful_val, cfg.template,
                                                 refusal_toks, cfg.prune_layer_pct,
                                                 cfg.batch_size, filtered=False)["bypass"])
        extra["control_bypass"] = np.mean(curves, axis=0)
        extra["control_curves"] = np.asarray(curves)
        logger.info("[%s] control peak=%.3f  vs  refusal peak=%.3f",
                    stage, float(np.nanmax(extra["control_bypass"])),
                    float(np.nanmax(res["bypass"])))

    os.makedirs(cfg.results_dir, exist_ok=True)
    path = f"{cfg.results_dir}/{stage}_refusal.npz"

    # np.savez rewrites the WHOLE file, so re-running one stage with fewer flags would
    # silently delete results from an earlier run (e.g. `--stage base` without --control
    # destroyed base's control_bypass). Carry forward any key this run did not recompute.
    # Safe because everything here is deterministic given (checkpoint, config, seed).
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        carried = [k for k in old.files
                   if k.startswith(("control_", "substring_", "sample_", "n_behavioral"))
                   and k not in extra]
        for k in carried:
            extra[k] = old[k]
        if carried:
            logger.info("[%s] carried forward from previous run: %s", stage, ", ".join(carried))
    np.savez(path, stage=np.array(stage), model_id=np.array(model_id),
             bypass=res["bypass"], l_star=np.array(res["l_star"]),
             baseline_refusal=np.array(res["baseline_refusal"]),
             excluded_layers=res["excluded_layers"], kl=res["kl"], steer=res["steer"],
             valid=res["valid"], pos_star=np.array(res["pos_star"]),
             naive_l_star=np.array(res["naive_l_star"]), **extra)
    kl_at = (float(res["kl"][res["pos_star"], res["l_star"]]) if valid_direction
             else float("nan"))
    logger.info("[%s] saved %s | l*=%d (naive %d) baseline_refusal=%.3f peak_strength=%.3f "
                "KL@l*=%.4f", stage, path, res["l_star"], res["naive_l_star"],
                res["baseline_refusal"], float(np.nanmax(res["bypass"])), kl_at)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="stage name (base/sft/dpo) or 'all'")
    ap.add_argument("--control", action="store_true",
                    help="also run the norm-matched random-direction negative control")
    ap.add_argument("--behavioral", action="store_true",
                    help="also measure substring refusal rate, baseline vs ablated")
    args = ap.parse_args()
    cfg = DEFAULT
    ckpts = dict(cfg.checkpoints)
    stages = list(ckpts) if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in ckpts:
            raise SystemExit(f"unknown stage '{s}'. known: {list(ckpts)}")
    for s in stages:
        run_one(s, ckpts[s], cfg, control=args.control, behavioral=args.behavioral)


if __name__ == "__main__":
    main()
