"""Diagnose WHAT the model actually emits at the first generated position.

    python diagnose_refusal_token.py --stage sft

Motivation: verify_setup.py proved the refusal string tokenizes CONSISTENTLY across
checkpoints. It did NOT prove the resulting id is the token the model actually produces.
The first run showed baseline_refusal ~= -9 (base) and ~= -11.7 (SFT) — both at the floor,
with SFT *lower* than base. That is the signature of scoring a token the model never emits,
not of a model that does not refuse.

This script answers three things directly, with no guessing:
  1. Top-k first-token distribution on harmful prompts -> is 315 ('I') anywhere near the top?
  2. Greedy continuations -> does the model refuse at all, and with what wording?
  3. Same two, with and without Zephyr's <|system|> turn -> does our template degrade it?

Cheap: one model, a handful of prompts, no sweep.
"""

from __future__ import annotations

import argparse
import logging

import torch

from config import DEFAULT, config_for
from data import assert_available, load_instructions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose")

def with_system_turn(template: str) -> str:
    """The same template with an (often empty) system turn prepended. Several chat formats
    expect one, and omitting it can degrade the model — worth testing as a second panel."""
    return "<|system|>\n</s>\n" + template

# Candidate refusal-onset strings. Refusals overwhelmingly open with one of these.
CANDIDATES = ["I", " I", "I'm", "As", "Sorry", "Unfortunately", "It"]


def report(model, tok, instructions, template, label: str, topk: int = 12) -> None:
    print(f"\n{'=' * 78}\n{label}\n  template={template!r}\n{'=' * 78}")

    prompts = [template.format(instruction=i) for i in instructions]
    enc = tok(prompts, padding=True, return_tensors="pt")
    ids = enc.input_ids.to(model.device)
    mask = enc.attention_mask.to(model.device)

    logits = model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    # 1. what the model actually wants to say first, aggregated over the batch
    mean_p = probs.mean(dim=0)
    top_p, top_i = mean_p.topk(topk)
    print("\n-- top first tokens (mean prob over prompts) --")
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        print(f"   id={i:<7d} {tok.decode([i])!r:<14} p={p:.4f}")

    # 2. where our candidates actually rank
    print("\n-- candidate refusal-onset tokens --")
    rank_of = {int(t): r for r, t in enumerate(mean_p.argsort(descending=True).tolist()[:2000])}
    for s in CANDIDATES:
        cid = tok.encode(s, add_special_tokens=False)
        if len(cid) != 1:
            print(f"   {s!r:<14} -> {cid} (not a single token, skipped)")
            continue
        t = cid[0]
        r = rank_of.get(t)
        print(f"   {s!r:<14} id={t:<7d} p={mean_p[t].item():.6f} "
              f"rank={'>2000' if r is None else r}")

    # 3. does it refuse at all
    print("\n-- greedy continuations (first 3 prompts) --")
    gen = model.generate(input_ids=ids[:3], attention_mask=mask[:3], max_new_tokens=40,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    for j in range(gen.shape[0]):
        out = tok.decode(gen[j, ids.shape[1]:], skip_special_tokens=True)
        print(f"   [{j}] {instructions[j][:60]!r}\n       -> {out.strip()[:160]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="zephyr",
                    help="model family from config.LINEAGES "
                         "(zephyr | olmo2 | tulu2)")
    ap.add_argument("--stage", default="sft", help="stage name from config.checkpoints")
    ap.add_argument("--n", type=int, default=8, help="# harmful prompts to average over")
    args = ap.parse_args()

    assert_available()               # before loading 15GB of weights
    cfg = config_for(args.lineage)   # deliberately NOT require_verified():
    ckpts = dict(cfg.checkpoints)    # this script is what makes a lineage verifiable
    if args.stage not in ckpts:
        raise SystemExit(f"unknown stage '{args.stage}'. known: {list(ckpts)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    model_id = ckpts[args.stage]
    logger.info("loading %s (%s)", model_id, cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, cfg.dtype), device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    harmful = load_instructions("harmful_val")[: args.n]
    report(model, tok, harmful, cfg.template,
           f"[{cfg.lineage}/{args.stage}] HARMFUL / lineage template")
    report(model, tok, harmful, with_system_turn(cfg.template),
           f"[{cfg.lineage}/{args.stage}] HARMFUL / template WITH <|system|> turn")

    harmless = load_instructions("harmless_val")[: args.n]
    report(model, tok, harmless, cfg.template,
           f"[{cfg.lineage}/{args.stage}] HARMLESS / lineage template (should NOT refuse)")

    print(f"\n{'=' * 78}\nNEXT: set Lineage.expected_refusal_id for '{cfg.lineage}' in "
          f"config.py to the TOP-1 id above\n(on HARMFUL prompts), then run "
          f"verify_setup.py --lineage {cfg.lineage} for the pinned n_eoi.\n{'=' * 78}")


if __name__ == "__main__":
    main()
