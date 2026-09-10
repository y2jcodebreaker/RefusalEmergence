# Refusal Emergence Across the Alignment Pipeline

**When does a language model learn to refuse — and where?** Is refusal already in the
**base** model, does **SFT** create it, or does the **preference-optimization step (DPO/RLHF)**
install it, and in which layers?

Prior work (Arditi, NeurIPS 2024) shows refusal is a single direction in *finished* chat
models. "How Post-Training Reshapes LLMs" (COLM 2025) compared base-vs-final (2 points).
**This maps the full developmental trajectory — base → SFT → DPO, per layer.**

Method reuses a **validated** Arditi refusal-direction implementation (reproduced Arditi's
`(pos=-5, layer=12)` exactly in a prior project).

## Run

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
export ARDITI_REPO=/path/to/refusal_direction     # cloned Arditi repo (for harmful/harmless splits)

python run_stage.py --stage all      # base, then sft, then dpo (one 7B at a time)
python aggregate.py                  # -> results/figures/refusal_emergence_{heatmap,peak}.pdf
```

## What each stage does
`run_stage.py` per checkpoint: extract the refusal direction at every layer (mean-diff
harmful−harmless at end-of-instruction tokens), then measure **causal** strength per layer
= how much ablating that layer's direction reduces refusal on harmful prompts.
`aggregate.py` stacks the three curves into a **stage × layer heatmap** + a per-stage peak panel.

## Models (one lineage)
`Mistral-7B-v0.1` (base) → `zephyr-7b-sft-full` (SFT) → `zephyr-7b-beta` (DPO). Same vocab,
so the fixed chat template tokenizes identically across all three (controlled comparison).

## MUST-VERIFY before trusting results (flagged in config.py)
- **Chat template** held fixed across stages; base Mistral has no native chat format — using
  the Zephyr template on all three is the controlled choice (weights are the only variable).
- **Refusal token** resolved per-tokenizer from `" I"` (Arditi used `'I'` for Llama-3).
  Confirm it's a sensible refusal-onset token for Mistral/Zephyr.
- **O-40:** the last 20% of layers are excluded for `l*` selection; the FULL curve is reported.

## Predicted result
Refusal weak/non-causal in **base**, emerging at **SFT** and/or sharpened by **DPO**,
localized to **early-middle layers** (a prior project found L11–14 on Llama-3.1).
Either way, the figure is the finding.
