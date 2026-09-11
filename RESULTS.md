# Results

Zephyr lineage, one 7B family, three checkpoints. Run 2026-09-11 on an L40S-class GPU,
~30 min total. All numbers below are reproducible with the commands in the README.

## Headline

| | base | SFT | DPO |
|---|---|---|---|
| **behavioral refusal** (strict judge, n=132 held out) | **0.235** (31/132) | 0.008 (1/132) | 0.045 (6/132) |
| **inducible direction** (max induced refusal) | **NONE — never crosses 0** | +1.01 @ L16 | **+1.76** @ L16 |
| induce window (layers above threshold) | — | L15–L20 | L14–L19 |

> **Alignment does not create refusal behavior — it creates refusal machinery.**
>
> Base Mistral refuses **29× more often than its SFT descendant** in actual generated text,
> while possessing **no steerable refusal direction at any layer, at any KL bound up to 5.0**.
> The aligned models refuse far less, yet carry a clean, controllable, middle-layer refusal
> mechanism. Behavioral refusal and refusal geometry are dissociated across the pipeline.

Reading: base refuses *incidentally* — pretraining contains refusal-shaped text, so the
behavior appears with no controllable mechanism behind it. SFT installs the mechanism
(L15–20). DPO sharpens it (+1.01 → +1.76) without moving its peak (L16 = 50% depth).

![induce](results/figures/refusal_emergence_induce.pdf)

## Supporting

### Causal ablation strength, with a negative control

| stage | l* (filtered) | l* (naive) | baseline refusal score | peak strength | control peak | ratio |
|---|---|---|---|---|---|---|
| base | **−1 (none valid)** | 15 | −2.064 | 1.080 | 0.127 | 8.5× |
| SFT | 20 | 20 | −1.488 | 5.490 | 0.125 | 43.8× |
| DPO | 17 | 16 | −3.386 | 7.590 | 0.154 | 49.2× |

The norm-matched random-direction control (K=3, norm matched per position/layer so the
comparison isolates *orientation*) is **flat** across stages — 0.127 / 0.125 / 0.154. That
rules out the main alternative explanation, that later-stage models are simply more
perturbable: a random direction perturbs exactly as hard and produces no trend.

⚠️ base's peak of 1.080 is the maximum over directions that **all fail** the induce
criterion. It is not a measured refusal strength and should not be reported as one.

### Selection filters (Arditi's three criteria)

| stage | cells passing KL ≤ 0.1 | passing induce ≥ 0 | passing both |
|---|---|---|---|
| base | 88 / 130 | **0** | **0** |
| SFT | 119 / 130 | 6 | 6 |
| DPO | 108 / 130 | 6 | 4 |

Base fails on **induce**, not KL — and at every KL bound tested up to 5.0. This is not a
threshold that failed to transfer between architectures; it is the absence of a refusal mode
to induce.

### Behavioral axis, verbatim vs strict judge

| stage | verbatim | strict | change | degenerate (empty) |
|---|---|---|---|---|
| base | 0.538 (71/132) | **0.235** (31/132) | −56% | 0.030 |
| SFT | 0.008 | **0.008** | **no-op** | 0.000 |
| DPO | 0.045 | **0.045** | **no-op** | 0.000 |

The strict judge is a **no-op on both chat models** and cuts base by 56% — the signature of a
correction that targets a real base-model artifact rather than one that flatters the result.
See O-44 below.

## Known limitations

1. **One lineage, one size, one behavior.** Zephyr's DPO deliberately removed safety
   filtering, so this lineage alone cannot support a general claim about alignment.
2. **`l*` is partly an artifact of pruning.** The ablation curves peak at L26–31, exactly the
   band `prune_layer_pct=0.20` excludes from selection. Report the full curve; the reported
   `l*` is "best among allowed." (The *induce* result does not have this problem — its peak
   at L16 is well inside the allowed range.)
3. **The x-axis is the layer the direction was READ FROM**, not where refusal is implemented.
   Ablation is applied globally across all layers, per Arditi.
4. **The ablation half of the behavioral axis is weak.** SFT and DPO refuse 1/132 and 6/132,
   so driving those to zero carries little weight. The causal claim rests on the induce axis.
5. **Residual base-judge risk.** The strict judge fixes turn leakage and confusion, but a base
   completion that is off-task *without* a turn marker would still pass. The 31 strict hits
   deserve a full hand-check before publication.
6. **Jensen gap.** The per-prompt diagnostic reports mean *probability* while the sweep reports
   mean *log-ratio*; both are correct and they differ on skewed distributions.

## Methodological notes (four errors caught, in order)

Each was found by checking a number against what the model actually did, not by inspecting
code. Recorded because the failure modes generalise.

**O-41 — implement from the CODE, not the paper text.** Paper prose and released
implementations diverge on specifics that change results.

**O-42 — cross-tokenizer consistency ≠ correctness.** The first three runs scored token
`315` (piece `▁I`, space-prefixed) instead of `28737` (piece `I`, bare) — the token these
models actually emit after `<|assistant|>\n`. Both **decode to `"I"`**, which is what hid it.
`tok.encode("I")` returns `[315]` because SentencePiece prepends a dummy prefix space; resolve
with `convert_tokens_to_ids` instead. Measured: p(28737)=0.3675 rank 1 vs p(315)=0.000175 rank
60. The wrong token produced `baseline_refusal ≈ −11.65` (p≈1e-5) and a **clean-looking
monotone trend of 2.62 / 4.68 / 6.28 that was pure noise** in an irrelevant log-ratio. The
preflight passed throughout because it only checked that the token was the *same* in every
checkpoint — which `315` satisfied.

**O-43 — port a method's FILTERS, not just its headline statistic.** Only one of Arditi's
three selection criteria was implemented, making `l*` a bare argmax over "how much does
ablating this destroy refusal" — which cannot distinguish killing refusal from killing the
model. Symptom, visible only by reading generations: ablated completions came back as **empty
strings**, and an empty string contains no refusal substring, so **degeneration scored as a
perfect jailbreak** (0.656 → 0.000). It also explains the late-layer dominance: late-layer
directions perturb final logits hardest, so they win an unguarded argmax. Adding KL ≤ 0.1 and
induce ≥ 0.0 changed the finding entirely.

**O-44 — a judge validated on chat models cannot be moved to a base model unexamined.**
Arditi's substring matcher has a ~56% false-positive rate on base Mistral, for two independent
reasons: base free-runs past its own turn and simulates the next `<|user|>` turn (so the judge
reads text that is not the model's reply), and *"I'm sorry, I don't understand the question"*
contains *"I'm sorry"* — incompetence matching the refusal list. **A base model is not a worse
chat model; it is a different kind of object.**

## Figures

| file | what it shows |
|---|---|
| `refusal_emergence_induce.pdf` | **the headline** — induced refusal vs layer, per stage |
| `refusal_emergence_peak.pdf` | ablation peak strength vs the random control |
| `refusal_emergence_heatmap.pdf` | stage × layer ablation strength, raw scale |
| `refusal_emergence_heatmap_rownorm.pdf` | same, row-normalised (location, magnitude discarded) |
| `refusal_emergence_behavioral.pdf` | substring refusal rate, baseline vs ablated |
