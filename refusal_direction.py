"""Refusal-direction extraction + per-layer causal strength — ported from the VALIDATED
Probe 4 (Arditi) in e01-depth-bands (which reproduced Arditi's (pos=-5, layer=12) exactly).

Faithful to Arditi: direction = mean(harmful) - mean(harmless) at resid_pre over end-of-
instruction tokens; directional ablation x -= (x . r_hat) r_hat at resid_pre + attn_out +
mlp_out, all layers; refusal_score = logP(refusal_tok) - logP(not) at last position.

Difference from E01: we run this per checkpoint and keep the per-layer causal-refusal-
strength curve (so aggregate.py can stack base/SFT/DPO into a stage x layer heatmap).

Pure helpers (refusal_score, select_l_star) are unit-tested in smoke_test.py without a model.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ pure helpers

def refusal_score(last_logits: torch.Tensor, refusal_toks: List[int], eps: float = 1e-8) -> torch.Tensor:
    """log P(refusal) - log P(non-refusal) at the last position. Arditi. (batch, vocab) -> (batch,)."""
    probs = torch.softmax(last_logits, dim=-1)
    refusal_p = probs[:, refusal_toks].sum(dim=-1)
    nonrefusal_p = (1.0 - refusal_p).clamp_min(0.0)
    return torch.log(refusal_p + eps) - torch.log(nonrefusal_p + eps)


def select_l_star(bypass_curve: np.ndarray, prune_pct: float) -> Tuple[int, Tuple[int, ...]]:
    """l* = argmax causal refusal strength among non-pruned layers (O-40: prune last 20%).
    Returns (l_star, pruned_layers). Full curve is reported regardless."""
    n = bypass_curve.shape[0]
    pruned = tuple(range(int(np.ceil(n * (1.0 - prune_pct))), n))
    masked = bypass_curve.copy()
    if pruned:
        masked[list(pruned)] = -np.inf
    l_star = int(np.nanargmax(masked))
    return l_star, pruned


# ------------------------------------------------------- model-dependent (GPU)

def _tokenize(tok, instructions: List[str], template: str):
    prompts = [template.format(instruction=i) for i in instructions]
    return tok(prompts, padding=True, truncation=False, return_tensors="pt")


def norm_matched_random(directions: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random directions with the per-(pos, layer) L2 norm matched to `directions`.

    Ported from E01 probe 7. Matching the norm is the point: it isolates ORIENTATION, so a
    difference cannot be explained by the control simply perturbing less hard."""
    norms = directions.norm(dim=-1, keepdim=True)                       # (pos, layer, 1)
    rand = torch.randn(directions.shape, generator=generator, dtype=torch.float32)
    rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)              # unit rows
    return (rand.to(directions.device) * norms).to(directions.dtype)


def resolve_refusal_token(tok, piece: str, expected_id: int | None = None) -> int:
    """Resolve a SentencePiece PIECE to its vocab id.

    Deliberately NOT tok.encode(): encode() prepends SentencePiece's dummy prefix space,
    so "I" silently becomes "_I" -> 315. The token a model actually emits right after
    "<|assistant|>\\n" has no preceding space, so it is the BARE piece "I" -> 28737.
    Both decode to "I", which is exactly what makes the mix-up invisible.
    Measured on zephyr-7b-beta, harmful prompts: p(28737)=0.3675 vs p(315)=0.000175."""
    tid = tok.convert_tokens_to_ids(piece)
    if tid is None or tid == tok.unk_token_id:
        raise ValueError(
            f"refusal piece {piece!r} is not in this tokenizer's vocab (got id={tid}). "
            f"Run diagnose_refusal_token.py to see what the model actually emits.")
    tid = int(tid)
    if expected_id is not None and tid != expected_id:
        raise ValueError(
            f"refusal piece {piece!r} -> id {tid}, but config expects {expected_id}. "
            f"The tokenizer changed; re-run diagnose_refusal_token.py before trusting "
            f"any result, and update Config.expected_refusal_id from its output.")
    return tid


def eoi_len(tok, template: str) -> int:
    """# tokens after {instruction} in the template = the end-of-instruction positions."""
    return len(tok.encode(template.split("{instruction}")[-1], add_special_tokens=False))


def get_mean_diff(model, tok, harmful, harmless, template, n_eoi, batch_size=16) -> torch.Tensor:
    """(n_eoi, n_layers, d) = mean harmful resid_pre - mean harmless, at eoi positions.
    forward_pre_hook on each block reads input[0] = resid_pre (Arditi generate_directions)."""
    layers = model.model.layers
    n_layers, d = len(layers), model.config.hidden_size
    positions = list(range(-n_eoi, 0))

    def mean_of(instructions):
        acc = torch.zeros((n_eoi, n_layers, d), dtype=torch.float64, device=model.device)
        n = len(instructions)
        handles = []
        for li, layer in enumerate(layers):
            def pre_hook(module, inp, li=li):
                a = inp[0] if isinstance(inp, tuple) else inp
                acc[:, li] += (1.0 / n) * a[:, positions, :].to(acc).sum(dim=0)
            handles.append(layer.register_forward_pre_hook(pre_hook))
        try:
            for i in range(0, n, batch_size):
                enc = _tokenize(tok, instructions[i:i + batch_size], template)
                model(input_ids=enc.input_ids.to(model.device),
                      attention_mask=enc.attention_mask.to(model.device))
        finally:
            for h in handles:
                h.remove()
        return acc

    return mean_of(harmful) - mean_of(harmless)


def _ablation_handles(model, direction: torch.Tensor):
    """x -= (x . r_hat) r_hat at resid_pre + attn_out + mlp_out, all layers (Arditi hook_utils)."""
    r = direction / (direction.norm() + 1e-8)

    def proj_pre(module, inp):
        a = inp[0] if isinstance(inp, tuple) else inp
        rr = r.to(a)
        a = a - (a @ rr).unsqueeze(-1) * rr
        return (a, *inp[1:]) if isinstance(inp, tuple) else a

    def proj_out(module, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        rr = r.to(a)
        a = a - (a @ rr).unsqueeze(-1) * rr
        return (a, *out[1:]) if isinstance(out, tuple) else a

    handles = []
    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(proj_pre))
        handles.append(layer.self_attn.register_forward_hook(proj_out))
        handles.append(layer.mlp.register_forward_hook(proj_out))
    return handles


def _last_logits(model, tok, instructions, template, batch_size=16) -> torch.Tensor:
    out = []
    for i in range(0, len(instructions), batch_size):
        enc = _tokenize(tok, instructions[i:i + batch_size], template)
        lg = model(input_ids=enc.input_ids.to(model.device),
                   attention_mask=enc.attention_mask.to(model.device)).logits[:, -1, :]
        out.append(lg.float().cpu())
    return torch.cat(out)


def _mean_refusal(logits, refusal_toks) -> float:
    return float(refusal_score(logits, refusal_toks).mean())


def refusal_strength_curve(model, tok, directions: torch.Tensor, harmful_val, template,
                           refusal_toks, prune_pct, batch_size=16):
    """Per-layer CAUSAL refusal strength for ONE checkpoint:
       strength[layer] = baseline_harmful_refusal - min_pos refusal_after_ablating(pos,layer).
    Higher = ablating that layer's direction kills refusal more = refusal concentrated there.
    Returns dict(bypass, l_star, baseline_refusal, excluded_layers)."""
    n_pos, n_layers, _ = directions.shape
    baseline = _mean_refusal(_last_logits(model, tok, harmful_val, template, batch_size), refusal_toks)

    abl = np.full((n_pos, n_layers), np.nan)
    for pos in range(n_pos):
        for layer in range(n_layers):
            h = _ablation_handles(model, directions[pos, layer])
            try:
                abl[pos, layer] = _mean_refusal(
                    _last_logits(model, tok, harmful_val, template, batch_size), refusal_toks)
            finally:
                for x in h:
                    x.remove()

    bypass = baseline - np.nanmin(abl, axis=0)     # (n_layers,)
    best_pos = np.nanargmin(abl, axis=0)           # (n_layers,) winning position per layer
    l_star, pruned = select_l_star(bypass, prune_pct)
    return {"bypass": bypass, "l_star": l_star, "baseline_refusal": baseline,
            "excluded_layers": np.array(pruned), "best_pos": best_pos}
