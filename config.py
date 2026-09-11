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
    #
    # MEASURED 2026-09-11 (diagnose_refusal_token.py, zephyr-7b-beta, harmful prompts):
    #   id=28737 piece 'I'   p=0.3675   <- rank 1, what the model ACTUALLY emits
    #   id=315   piece '_I'  p=0.000175 <- rank 60
    # Control on harmless prompts: p(28737)=0.0027 -> a 135x harmful/harmless contrast.
    #
    # The two pieces both DECODE to "I", which is what made this hard to see:
    #   315   = "_I" (word-initial, space-prefixed)
    #   28737 = "I"  (bare, no space prefix)   <- correct after "<|assistant|>\n"
    # Resolve with convert_tokens_to_ids, NOT encode(): tok.encode("I") returns [315]
    # because SentencePiece prepends a dummy prefix space, silently turning "I" into "_I".
    # Scoring 315 gave baseline_refusal ~= -11.65 (p ~= 1e-5) and a clean-looking but
    # meaningless monotone trend across stages.
    refusal_token_piece: str = "I"
    # Cross-check: the piece must resolve to this id. Guards against a tokenizer swap
    # silently changing which token is scored. Update only with a fresh diagnostic run.
    expected_refusal_id: int = 28737
    # Pinned so every stage uses the SAME end-of-instruction position window (see N_EOI_FIXED).
    n_eoi: int = N_EOI_FIXED
    n_train: int = 128       # samples for the mean-diff direction (Arditi default)
    n_val: int = 32          # samples for the per-layer causal sweep
    prune_layer_pct: float = 0.20   # O-40: mark last 20% of layers as excluded for l* (report full curve)
    # Arditi's OTHER TWO selection criteria (select_direction.py; values verified in E01,
    # which reproduced Arditi's (pos=-5, layer=12) exactly).
    # kl_threshold is the load-bearing one: without it, l* selection is a pure argmax over
    # "how much does ablating this destroy refusal", which happily picks the direction that
    # destroys the MODEL -- a lobotomised model emits no refusal token either. The first
    # behavioral run exposed this: ablated generations came back as EMPTY STRINGS, which the
    # substring judge scored as "complied", i.e. a fake 100% jailbreak.
    kl_threshold: float = 0.1        # max KL on harmless prompts after ablation
    induce_threshold: float = 0.0    # min refusal on harmless after ADDING the direction
    # Negative control: K norm-matched RANDOM directions through the identical sweep.
    # Rules out "later-stage models are just more perturbable" as the reason peak strength
    # rises across stages. If the control curve climbs too, the headline finding is dead.
    n_control: int = 3
    # Behavioral axis (--behavioral): greedy generation length for substring refusal matching.
    # 48 comfortably covers Arditi's refusal prefixes without paying for long completions.
    gen_max_new_tokens: int = 48
    # Behavioral n is DECOUPLED from n_val. The causal sweep costs n_pos*n_layers forward
    # passes so n_val stays small, but a RATE at n=32 is quantised to 1/32=0.031 -- the first
    # run put SFT's entire refusal rate on a single completion. Generation is cheap by
    # comparison (2 passes), so use every harmful_val example available.
    # 0 = use the whole held-out tail harmful_train[n_train:] (132 prompts, touched by
    # neither direction fitting nor l* selection). harmful_val has only 39 and its head
    # drives l*, so it is both too small and not fully clean for a behavioral rate.
    n_behavioral: int = 0
    n_sample_completions: int = 8    # how many to store per condition for eyeballing
    batch_size: int = 16
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "results/figures"


DEFAULT = Config()
