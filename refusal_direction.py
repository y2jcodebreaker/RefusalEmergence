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
    """UNFILTERED argmax — kept only for the smoke test and for reporting what a naive
    selection would have picked. DO NOT use for real selection: it happily chooses the
    direction that destroys the model (see select_direction_arditi)."""
    n = bypass_curve.shape[0]
    pruned = tuple(range(int(np.ceil(n * (1.0 - prune_pct))), n))
    masked = bypass_curve.copy()
    if pruned:
        masked[list(pruned)] = -np.inf
    l_star = int(np.nanargmax(masked))
    return l_star, pruned


def select_direction_arditi(ablation_refusal: np.ndarray, steering_refusal: np.ndarray,
                            kl_harmless: np.ndarray, baseline_harmful: float,
                            kl_threshold: float, induce_threshold: float, prune_pct: float):
    """Arditi's FULL selection — three criteria, not one (select_direction.py).

    Ported from E01 probe 4 (which reproduced Arditi's (pos=-5, layer=12) exactly).
      1. bypass  — ablation lowers refusal on harmful             (minimise ablation_refusal)
      2. induce  — ADDING the direction raises refusal on harmless (>= induce_threshold)
      3. KL      — ablation barely perturbs harmless behaviour     (<= kl_threshold)

    Criterion 3 is the one that matters here. Without it the argmax picks whatever direction
    damages the network most, because a broken model also stops emitting the refusal token.
    Observed symptom of its absence: ablated generations came back as EMPTY STRINGS, which a
    substring judge scores as "complied" — a fake jailbreak.

    Returns (bypass_strength, l_star, pos_star, pruned, valid_mask). l_star = -1 when NO
    (pos, layer) passes the filters: that is a real answer, not an error."""
    n_pos, n_layers = ablation_refusal.shape
    best_pos = np.nanargmin(ablation_refusal, axis=0)
    bypass_strength = baseline_harmful - ablation_refusal[best_pos, np.arange(n_layers)]

    pruned = tuple(range(int(np.ceil(n_layers * (1.0 - prune_pct))), n_layers))

    valid = ~np.isnan(ablation_refusal)
    valid &= kl_harmless <= kl_threshold
    valid &= steering_refusal >= induce_threshold
    if pruned:
        valid[:, list(pruned)] = False
    if not valid.any():
        return bypass_strength, -1, -1, pruned, valid

    masked = np.where(valid, ablation_refusal, np.inf)
    pos_star, l_star = (int(v) for v in np.unravel_index(np.argmin(masked), masked.shape))
    return bypass_strength, l_star, pos_star, pruned, valid


def kl_last(baseline: torch.Tensor, intervention: torch.Tensor, eps: float = 1e-6) -> float:
    """KL(baseline || intervention) over last-position logits, float64. Arditi kl_div_fn."""
    a = baseline.to(torch.float64).softmax(dim=-1)
    b = intervention.to(torch.float64).softmax(dim=-1)
    return float((a * (torch.log(a + eps) - torch.log(b + eps))).sum(dim=-1).mean())


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


def transformer_layers(model) -> "torch.nn.ModuleList":
    """The decoder-block ModuleList, however the model happens to be wrapped.

    Every hook in this repo needs this one object. `model.model.layers` is correct for a
    plain HF causal LM and WRONG for a peft-wrapped one: PeftModel forwards attribute access
    to its base_model, so `model.model` lands on the *ForCausalLM rather than the inner
    *Model, and `.layers` raises. That cost the first dose_response run
    (AttributeError: 'Olmo2ForCausalLM' object has no attribute 'layers', 2026-09-22) --
    every measurement in this repo had only ever been called on an unwrapped model, so the
    assumption held by accident for months.

    Rather than hardcode a wrapper depth, this tries the known layouts and then falls back to
    finding the longest ModuleList in the module tree, which is the decoder stack in every
    decoder-only architecture. LoRA adapters live in ModuleDicts keyed by adapter name, not
    in a long ModuleList, so they cannot be mistaken for it.
    """
    for path in (("model", "layers"),                                  # plain HF
                 ("base_model", "model", "model", "layers"),           # peft
                 ("transformer", "h"),                                 # gpt2-style
                 ("layers",)):
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, torch.nn.ModuleList) and len(node) > 1:
            return node

    best = max((m for _, m in model.named_modules()
                if isinstance(m, torch.nn.ModuleList) and len(m) > 1),
               key=len, default=None)
    if best is not None:
        logger.warning("transformer_layers: no known layout matched %s; using the longest "
                       "ModuleList (%d entries). Verify this is the decoder stack.",
                       type(model).__name__, len(best))
        return best
    raise SystemExit(
        f"cannot locate the decoder-block ModuleList on {type(model).__name__}. Every hook "
        f"in this repo needs it. Top-level submodules: "
        f"{[n for n, _ in model.named_children()][:12]}")


def get_mean_diff(model, tok, harmful, harmless, template, n_eoi, batch_size=16) -> torch.Tensor:
    """(n_eoi, n_layers, d) = mean harmful resid_pre - mean harmless, at eoi positions.
    forward_pre_hook on each block reads input[0] = resid_pre (Arditi generate_directions)."""
    layers = transformer_layers(model)
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
    for layer in transformer_layers(model):
        handles.append(layer.register_forward_pre_hook(proj_pre))
        handles.append(layer.self_attn.register_forward_hook(proj_out))
        handles.append(layer.mlp.register_forward_hook(proj_out))
    return handles


def _addition_handles(model, vector: torch.Tensor, coeff: float, layer: int):
    """Activation addition: add coeff*vector to the source layer's block INPUT only.
    RAW (un-normalised) mean-diff vector with coeff=1.0, per Arditi select_direction.py."""
    v = vector.detach()

    def add_pre(module, inp):
        a = inp[0] if isinstance(inp, tuple) else inp
        a = a + coeff * v.to(a)
        return (a, *inp[1:]) if isinstance(inp, tuple) else a

    return [transformer_layers(model)[layer].register_forward_pre_hook(add_pre)]


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
                           refusal_toks, prune_pct, batch_size=16, harmless_val=None,
                           kl_threshold=0.1, induce_threshold=0.0, filtered=True):
    """Per-layer CAUSAL refusal strength for ONE checkpoint, with Arditi's full selection.

    strength[layer] = baseline_harmful_refusal - min_pos refusal_after_ablating(pos, layer).

    When `harmless_val` is given (filtered=True), also measures the KL side-effect of each
    ablation and the induce effect of each addition, and selects (pos*, l*) under all three
    of Arditi's criteria. Without the KL filter the argmax picks model-destroying directions.
    Pass filtered=False (the control path) to skip the two extra sweeps."""
    n_pos, n_layers, _ = directions.shape
    baseline = _mean_refusal(_last_logits(model, tok, harmful_val, template, batch_size), refusal_toks)

    do_filter = filtered and harmless_val is not None
    base_harmless = (_last_logits(model, tok, harmless_val, template, batch_size)
                     if do_filter else None)

    abl = np.full((n_pos, n_layers), np.nan)
    steer = np.full((n_pos, n_layers), np.nan)
    kl = np.full((n_pos, n_layers), np.nan)
    for pos in range(n_pos):
        for layer in range(n_layers):
            d = directions[pos, layer]
            h = _ablation_handles(model, d)
            try:
                abl[pos, layer] = _mean_refusal(
                    _last_logits(model, tok, harmful_val, template, batch_size), refusal_toks)
                if do_filter:
                    kl[pos, layer] = kl_last(
                        base_harmless,
                        _last_logits(model, tok, harmless_val, template, batch_size))
            finally:
                for x in h:
                    x.remove()
            if do_filter:
                h = _addition_handles(model, d, coeff=1.0, layer=layer)
                try:
                    steer[pos, layer] = _mean_refusal(
                        _last_logits(model, tok, harmless_val, template, batch_size), refusal_toks)
                finally:
                    for x in h:
                        x.remove()

    bypass = baseline - np.nanmin(abl, axis=0)     # (n_layers,)
    best_pos = np.nanargmin(abl, axis=0)           # (n_layers,) winning position per layer

    if do_filter:
        bypass, l_star, pos_star, pruned, valid = select_direction_arditi(
            abl, steer, kl, baseline, kl_threshold, induce_threshold, prune_pct)
        naive_l, _ = select_l_star(bypass, prune_pct)
        if l_star < 0:
            # WHICH criterion killed it, and by how much. Never just report "none passed":
            # "no direction survives the KL bound" and "no direction induces refusal" are
            # completely different claims about the model.
            unpruned = np.ones_like(kl, dtype=bool)
            if pruned:
                unpruned[:, list(pruned)] = False
            kl_ok = (kl <= kl_threshold) & unpruned
            ind_ok = (steer >= induce_threshold) & unpruned
            logger.warning(
                "NO (pos, layer) passes Arditi's filters. Breakdown over %d unpruned cells:\n"
                "    KL <= %.2f      : %d pass  (min KL observed %.4f, median %.4f)\n"
                "    induce >= %.2f  : %d pass  (max steer observed %.4f, median %.4f)\n"
                "    BOTH            : %d pass\n"
                "  The unfiltered argmax would have picked layer %d.",
                int(unpruned.sum()), kl_threshold, int(kl_ok.sum()),
                float(np.nanmin(kl[unpruned])), float(np.nanmedian(kl[unpruned])),
                induce_threshold, int(ind_ok.sum()),
                float(np.nanmax(steer[unpruned])), float(np.nanmedian(steer[unpruned])),
                int((kl_ok & ind_ok).sum()), naive_l)
        else:
            logger.info("l*=%d pos*=%d (Arditi-filtered) | unfiltered argmax would be %d | "
                        "%d/%d (pos,layer) cells pass", l_star, pos_star - n_pos, naive_l,
                        int(valid.sum()), valid.size)
        return {"bypass": bypass, "l_star": l_star, "baseline_refusal": baseline,
                "excluded_layers": np.array(pruned), "best_pos": best_pos,
                "pos_star": pos_star, "kl": kl, "steer": steer, "valid": valid,
                "naive_l_star": naive_l}

    l_star, pruned = select_l_star(bypass, prune_pct)
    return {"bypass": bypass, "l_star": l_star, "baseline_refusal": baseline,
            "excluded_layers": np.array(pruned), "best_pos": best_pos}
