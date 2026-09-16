"""P1-E1c — does a transplanted direction produce REFUSAL TEXT, or just wreck the logits?

    python transplant_text.py --lineage olmo2 --target base --source sft

transplant.py answers "does the refusal token's log-odds cross zero". On OLMo 2 it crosses
only at an injected norm where KL(last-token) reaches 6-8, which is enormous next to the 0.1
Arditi allows on the ablation side. A reviewer's first question is therefore unavoidable:
did the injection install refusal, or did it break the model into emitting *something* whose
first token happens to be "I"?

A logit cannot answer that. Text can. This script adds the source direction to the target at
the coefficients transplant.py already swept, GENERATES, and reports:

  * the substring refusal rate (Arditi/JailbreakBench prefixes), strict and verbatim,
  * the DEGENERATE rate (empty, or one token repeated) — the "broken, not refusing" signal,
  * the same for a norm-matched RANDOM direction at the identical coefficient.

The random arm is the control that decides it. If real and random are equally degenerate but
only real refuses, the induction is real. If both degenerate and neither refuses in text, the
logit crossing was distributional damage and the transplant claim does not survive.

Prompts are HARMLESS (the induction question is "does refusal appear where it should not"),
matching transplant.py's evaluation set exactly.
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
from refusal_direction import _addition_handles, resolve_refusal_token
from refusal_substring import (generate_completions, refusal_rate, truncate_at_turn)
from runlog import RunRecord
from transplant import coeff_grid, load_source_directions, set_seed, source_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("P1-E1c")

EXPERIMENT = "P1-E1c"
QUESTION = ("When a transplanted direction crosses the induction threshold, is the output "
            "coherent refusal text or distributional damage?")


def degenerate_rate(completions: list[str]) -> float:
    """Fraction that are empty, or a single token repeated to the generation limit.

    Both are what a wrecked model emits. Neither is a refusal, and neither is a compliance --
    scoring them as 'complied' is exactly how an earlier run reported a fake jailbreak."""
    if not completions:
        return float("nan")
    n_bad = 0
    for c in completions:
        t = truncate_at_turn(c).strip()
        if len(t) < 2:
            n_bad += 1
            continue
        words = t.split()
        # >=8 words and <=2 distinct: "I I I I ..." or "the the the ...".
        if len(words) >= 8 and len(set(w.lower() for w in words)) <= 2:
            n_bad += 1
    return n_bad / len(completions)


def arm(model, tok, prompts, template, direction, coeff, layer, cfg) -> dict:
    handles = _addition_handles(model, direction, coeff=coeff, layer=layer)
    try:
        comps = generate_completions(model, tok, prompts, template,
                                     cfg.gen_max_new_tokens, cfg.batch_size)
    finally:
        for h in handles:
            h.remove()
    return {"refusal": refusal_rate(comps),
            "refusal_strict": refusal_rate(comps, strict=True),
            "degenerate": degenerate_rate(comps),
            "completions": comps}


def _norm_matched(d: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r = torch.randn(d.shape, generator=gen, dtype=torch.float32)
    return (r / (r.norm() + 1e-8) * d.norm().cpu()).to(d.device).to(d.dtype)


def crossing_coeffs(cfg, target: str, source: str) -> list[float]:
    """The coefficients transplant.py found interesting for this cell: the FIRST one that
    crossed the induction threshold, and the one with the largest induced score. Read from
    the saved sweep so this script and transplant.py can never disagree about the operating
    point (hardcoding it is how two scripts drift apart)."""
    path = cfg.path(target, "transplant")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run transplant.py --lineage {cfg.lineage} "
                         f"--stage {target} first")
    d = np.load(path, allow_pickle=True)
    cells, sweep = d["cells"], d["sweep"]          # 'src|layer|kind', (coeff, induced, kl)
    rows = [(float(s[0]), float(s[1])) for c, s in zip(cells, sweep)
            if str(c).split("|")[0] == source and str(c).split("|")[2] == "direction"]
    if not rows:
        raise SystemExit(f"no direction cells for source '{source}' in {path}")
    crossed = sorted(c for c, ref in rows if ref >= cfg.induce_threshold)
    best = max(rows, key=lambda r: r[1])[0]
    out = sorted({crossed[0], best} if crossed else {best})
    if not crossed:
        logger.warning("[%s<-%s] no coefficient crossed the threshold in transplant.py; "
                       "generating at the argmax (%.1f) anyway so the text is on record.",
                       target, source, best)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="olmo2")
    ap.add_argument("--target", required=True, help="model receiving the direction")
    ap.add_argument("--source", required=True, help="stage the direction came from")
    ap.add_argument("--n", type=int, default=64, help="harmless prompts to generate on")
    ap.add_argument("--coeff", type=float, default=None,
                    help="override; default = the crossing and argmax coefficients from "
                         "transplant.py's saved sweep")
    # BooleanOptionalAction, not store_true: with `action="store_true", default=True` the
    # flag can never be turned OFF, so a raw-norm transplant run could not be matched.
    ap.add_argument("--unit-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="must MATCH the transplant.py run being read (default: on; pass "
                         "--no-unit-norm to read a raw-norm run)")
    args = ap.parse_args()

    cfg = config_for(args.lineage)
    cfg.require_verified()
    logger.info("data: %s", assert_available())

    srcs, raw_norms = load_source_directions(cfg, source_layers(cfg), unit_norm=args.unit_norm)
    key = next((k for k in srcs if k[0] == args.source), None)
    if key is None:
        raise SystemExit(f"unknown source '{args.source}'. known: {[k[0] for k in srcs]}")
    _, layer, _pos = key
    coeffs = [args.coeff] if args.coeff else crossing_coeffs(cfg, args.target, args.source)
    logger.info("cell %s <- %s @L%d | raw_norm=%.1f | coefficients %s (grid was %s)",
                args.target, args.source, layer, raw_norms[key],
                [round(c, 1) for c in coeffs],
                [round(c, 1) for c in coeff_grid(raw_norms, args.unit_norm)])

    from run_stage import load_model
    set_seed(cfg.seed)
    model, tok = load_model(dict(cfg.checkpoints)[args.target], cfg.dtype)
    template, want_id, _n, is_ov = cfg.regime(args.target)
    if is_ov:
        logger.warning("[%s] REGIME OVERRIDE: generating under %r", args.target, template)
    else:
        resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)   # assert consistency
    prompts = load_instructions("harmless_val")[: args.n]
    vec = torch.from_numpy(srcs[key]).to(model.device)
    gen = torch.Generator().manual_seed(cfg.seed)
    rand = _norm_matched(vec, gen)

    with RunRecord(EXPERIMENT, "transplant_text.py", cfg, question=QUESTION) as rec:
        # coeff 0 is the untouched model: the floor every other row is read against.
        rows = {("none", 0.0): arm(model, tok, prompts, template, vec, 0.0, layer, cfg)}
        for c in coeffs:
            rows[("direction", c)] = arm(model, tok, prompts, template, vec, c, layer, cfg)
            rows[("random", c)] = arm(model, tok, prompts, template, rand, c, layer, cfg)

        print(f"\n{'=' * 84}\nP1-E1c  {args.target} <- {args.source} @L{layer}  "
              f"(n={len(prompts)} HARMLESS prompts)\n{'=' * 84}")
        print(f"{'arm':<10}{'coeff':>8}{'refusal':>10}{'strict':>9}{'degenerate':>13}")
        for (kind, c), r in rows.items():
            print(f"{kind:<10}{c:>8.1f}{r['refusal']:>10.3f}{r['refusal_strict']:>9.3f}"
                  f"{r['degenerate']:>13.3f}")
            rec.result(target=args.target, source=args.source, layer=layer, arm=kind, coeff=c,
                       refusal=round(r["refusal"], 4),
                       refusal_strict=round(r["refusal_strict"], 4),
                       degenerate=round(r["degenerate"], 4))

        out = f"{cfg.results_dir}/{cfg.lineage}_{args.target}_from_{args.source}_text.json"
        with open(out, "w") as f:
            json.dump({f"{k}|{c}": v for (k, c), v in rows.items()}, f, indent=1)
        print(f"\nwrote {out}")

        top = max((c for _, c in rows if c > 0), default=0.0)
        d_ref = rows[("direction", top)]["refusal_strict"] if ("direction", top) in rows else 0
        r_ref = rows[("random", top)]["refusal_strict"] if ("random", top) in rows else 0
        d_deg = rows[("direction", top)]["degenerate"] if ("direction", top) in rows else 0
        print("\n-- sample completions (direction arm, highest coefficient) --")
        for c in rows.get(("direction", top), {}).get("completions", [])[:6]:
            print("   " + truncate_at_turn(c).strip().replace("\n", " ")[:110])
        print("\nVERDICT")
        if d_deg > 0.5:
            print("  -> DEGENERATE. The injection breaks generation; the logit crossing in\n"
                  "     transplant.py is distributional damage, not induced refusal. Report\n"
                  "     the cell as UNINTERPRETABLE at this coefficient.")
        elif d_ref > max(0.3, 3 * r_ref):
            print("  -> INDUCED REFUSAL IS BEHAVIOURAL. The target writes refusals it does not\n"
                  "     write untouched, a norm-matched random direction at the same\n"
                  "     coefficient does not, and the text is coherent.")
        else:
            print("  -> NOT SEPARATED FROM THE RANDOM CONTROL in text. The logit crossing does\n"
                  "     not reproduce behaviourally. Do not report this cell as induction.")
        print("=" * 84)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
