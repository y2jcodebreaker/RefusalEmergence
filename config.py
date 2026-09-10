"""Config for E02 — Refusal Emergence Across the Alignment Pipeline.

Everything a run needs, in one frozen place. The per-model bits (chat template,
refusal token) are MUST-VERIFY before trusting results — flagged inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Config", "DEFAULT"]

# Fixed chat template applied to ALL checkpoints (controlled experiment: hold prompt
# format constant, vary only the weights). base/SFT/DPO here all share the Mistral vocab,
# so this literal string tokenizes identically across the three. VERIFY if you change models.
ZEPHYR_TEMPLATE = "<|user|>\n{instruction}</s>\n<|assistant|>\n"


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
    # Refusals commonly start with "I" (Arditi used [40]='I' for Llama-3). Here we resolve
    # per-tokenizer from this string. VERIFY it's a sensible refusal-onset token per model.
    refusal_onset_str: str = " I"
    n_train: int = 128       # samples for the mean-diff direction (Arditi default)
    n_val: int = 32          # samples for the per-layer causal sweep
    prune_layer_pct: float = 0.20   # O-40: mark last 20% of layers as excluded for l* (report full curve)
    batch_size: int = 16
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "results/figures"


DEFAULT = Config()
