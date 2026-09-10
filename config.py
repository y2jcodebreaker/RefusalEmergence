"""Config for E02 — Refusal Emergence Across the Alignment Pipeline.

Everything a run needs, in one frozen place. The per-model bits (chat template,
refusal token) are MUST-VERIFY before trusting results — flagged inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Config", "DEFAULT"]

# Fixed chat template applied to ALL checkpoints (controlled experiment: hold prompt
# format constant, vary only the weights). All three share the Mistral 32000-token vocab.
# VERIFIED 2026-09-10 (verify_setup.py): base Mistral has NO native chat template, so
# imposing the Zephyr format on all three is the correct controlled choice.
ZEPHYR_TEMPLATE = "<|user|>\n{instruction}</s>\n<|assistant|>\n"

# VERIFIED BUG FIX: tokenizing the template suffix gives eoi_len 9 (base) vs 10 (SFT/DPO)
# — the Zephyr tokenizers insert a phantom '' token (28705). A differing position-window
# size across stages would invalidate the stage-to-stage comparison, so we PIN n_eoi.
# 5 covers "<|assistant|>\n" (['|','ass','istant','|','>'] region) identically in all three.
N_EOI_FIXED = 5


@dataclass(frozen=True)
class Config:
    # (stage_name, hf_model_id) — a real base -> SFT -> DPO lineage.
    checkpoints: tuple[tuple[str, str], ...] = (
        ("base", "mistralai/Mistral-7B-v0.1"),
        ("sft",  "alignment-handbook/zephyr-7b-sft-full"),
        ("dpo",  "HuggingFaceH4/zephyr-7b-beta"),
    )
    dtype: str = "bfloat16"
    template: str = ZEPHYR_TEMPLATE
    # Refusal score = logP(refusal_tok) - logP(not), at the first generated position.
    # Refusals commonly start with "I" (Arditi used [40]='I' for Llama-3).
    # VERIFIED 2026-09-10: bare "I" -> [315] in ALL THREE tokenizers (base/SFT/DPO).
    # Do NOT use " I" (leading space): base -> [315] but SFT/DPO -> [28705, 315]
    # (phantom '' token), which is inconsistent across stages.
    refusal_onset_str: str = "I"
    # Pinned so every stage uses the SAME end-of-instruction position window (see N_EOI_FIXED).
    n_eoi: int = N_EOI_FIXED
    n_train: int = 128       # samples for the mean-diff direction (Arditi default)
    n_val: int = 32          # samples for the per-layer causal sweep
    prune_layer_pct: float = 0.20   # O-40: mark last 20% of layers as excluded for l* (report full curve)
    batch_size: int = 16
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "results/figures"


DEFAULT = Config()
