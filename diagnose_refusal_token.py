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

from config import config_for
from data import assert_available, load_instructions
from refusal_direction import _tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose")

def with_system_turn(tok, template: str) -> str | None:
    """The lineage's own template rendered WITH an empty system turn, or None if the
    tokenizer has no chat template to ask.

    Derived, never hand-built. An earlier version returned a hardcoded
    "<|system|>\n</s>\n" + template, which injects Mistral's </s> into any other family's
    prompt. On OLMo 2 that malformed prompt pushed <|endoftext|> to rank 5, dropped
    p(refusal) 0.948 -> 0.849, and flipped one harmful prompt from refusing to COMPLYING --
    i.e. the panel measured the broken prompt, not a design choice."""
    if not getattr(tok, "chat_template", None):
        return None
    try:
        rendered = tok.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": "\x00"}],
            tokenize=False, add_generation_prompt=True)
    except Exception:          # noqa: BLE001 - many templates reject an empty system turn
        return None
    out = rendered.replace("\x00", "{instruction}")
    bos = getattr(tok, "bos_token", None)
    if bos and out.startswith(bos):
        out = out[len(bos):]
    return out

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
        # convert_tokens_to_ids, NOT encode. encode() prepends SentencePiece's dummy prefix
        # space, so "I" becomes "_I" and resolves to a different id -- on tulu-2-dpo that is
        # 306 (p=3e-6, rank 94) against the 29902 (p=0.64) the model actually emits. This
        # panel used encode() and therefore displayed an id the rest of the codebase
        # deliberately avoids, which on 2026-09-22 made a CORRECT config look broken. The
        # resolver was always right; the diagnostic was lying to it.
        tid = tok.convert_tokens_to_ids(s)
        enc = tok.encode(s, add_special_tokens=False)
        if tid is None or tid == tok.unk_token_id:
            print(f"   {s!r:<14} not a single vocab piece (encode gives {enc}), skipped")
            continue
        t = int(tid)
        r = rank_of.get(t)
        warn = "  <- encode() would give " + str(enc[0]) if len(enc) == 1 and enc[0] != t else ""
        print(f"   {s!r:<14} id={t:<7d} p={mean_p[t].item():.6f} "
              f"rank={'>2000' if r is None else r}{warn}")

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
    ap.add_argument("--template", default=None,
                    help="override the lineage template (must contain {instruction}). Use to "
                         "test whether a BASE model that degenerates under the chat format is "
                         "coherent under a plain one — see O-56.")
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

    tpl = args.template or cfg.template
    if args.template:
        if "{instruction}" not in args.template:
            raise SystemExit("--template must contain {instruction}")
        logger.warning("TEMPLATE OVERRIDE %r — results are NOT comparable to runs using the "
                       "lineage template. This is a diagnostic, not a config change.", tpl)

    harmful = load_instructions("harmful_val")[: args.n]
    report(model, tok, harmful, tpl,
           f"[{cfg.lineage}/{args.stage}] HARMFUL / lineage template")
    sys_tpl = None if args.template else with_system_turn(tok, cfg.template)
    if sys_tpl and sys_tpl != cfg.template:
        report(model, tok, harmful, sys_tpl,
               f"[{cfg.lineage}/{args.stage}] HARMFUL / template WITH system turn (derived)")
    else:
        print(f"\n(skipping the system-turn panel: {args.stage}'s tokenizer offers no "
              f"distinct system-turn rendering)")

    harmless = load_instructions("harmless_val")[: args.n]
    report(model, tok, harmless, tpl,
           f"[{cfg.lineage}/{args.stage}] HARMLESS / lineage template (should NOT refuse)")

    # --- the exact config lines, computed rather than described.
    #
    # Reporting only the id is not enough, and that cost a step on 2026-09-22. The id the
    # model emits and the id `convert_tokens_to_ids(piece)` returns can DIFFER while both
    # decode to the same string: tulu-2-dpo emits id 29902 for "I" with p=0.64, while the
    # piece lookup for "I" returns 306, whose probability is 0.000003. So the PIECE STRING
    # has to be reported too -- it is the thing config stores, and it is not always the
    # character you expect. `convert_ids_to_tokens` is the only way to get it right.
    from refusal_direction import eoi_len as _eoi_len

    import torch as _t
    with _t.no_grad():
        enc = _tokenize(tok, load_instructions("harmful_train")[: args.n], tpl)
        top = model(input_ids=enc.input_ids.to(model.device),
                    attention_mask=enc.attention_mask.to(model.device)
                    ).logits[:, -1, :].softmax(-1).mean(0).argmax().item()
    piece = tok.convert_ids_to_tokens(int(top))
    round_trip = tok.convert_tokens_to_ids(piece)
    derived = _eoi_len(tok, tpl)

    print(f"\n{'=' * 78}\nPIN THESE IN config.LINEAGES[{cfg.lineage!r}]:\n")
    print(f'    refusal_token_piece={piece!r},')
    print(f"    expected_refusal_id={int(top)},")
    print(f"    n_eoi=<= {derived},        # tokenizer-derived eoi_len for this template")
    print()
    if round_trip != top:
        print(f"  ⚠️  convert_tokens_to_ids({piece!r}) = {round_trip}, NOT {top}. The piece "
              f"string does\n      not round-trip on this tokenizer, so "
              f"resolve_refusal_token would raise. Use the\n      piece printed above "
              f"verbatim -- it came from convert_ids_to_tokens and is exact.")
    else:
        print(f"  round-trip OK: {piece!r} -> {round_trip}")
    print(f"\n  n_eoi must be <= the SHORTEST derived eoi_len across this lineage's stages,\n"
          f"  and small enough that the window carries no instruction text -- "
          f"verify_setup.py\n  checks the second part against real prompts.\n{'=' * 78}")


if __name__ == "__main__":
    main()
