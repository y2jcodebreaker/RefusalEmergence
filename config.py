"""Config — one frozen place per LINEAGE, plus the run knobs shared across them.

    from config import DEFAULT, config_for
    cfg = config_for("olmo2")

A "lineage" is one model family's alignment chain (base -> SFT -> DPO -> ...). Three things
differ across families and **none of them transfers**:

  * the chat template,
  * which token id the model emits to open a refusal,
  * how many end-of-instruction positions that template occupies.

O-42 cost three runs to a wrong refusal token whose only symptom was a clean-looking trend.
So `expected_refusal_id` and `n_eoi` are **required per lineage**: a lineage that leaves them
None cannot be run until diagnose_refusal_token.py and verify_setup.py have been executed
against it. That is the lesson made structural rather than a comment nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["Config", "Lineage", "LINEAGES", "DEFAULT", "config_for"]

# ---------------------------------------------------------------- chat templates

# Zephyr. VERIFIED 2026-09-10: base Mistral has NO native chat template, so imposing this on
# all three checkpoints is the correct controlled choice (weights are the only variable).
ZEPHYR_TEMPLATE = "<|user|>\n{instruction}</s>\n<|assistant|>\n"

# OLMo 2 / Tulu. ⚠️ UNVERIFIED — confirm against the checkpoint's own tokenizer_config
# chat_template before trusting any number from these lineages.
OLMO2_TEMPLATE = "<|user|>\n{instruction}\n<|assistant|>\n"
TULU_TEMPLATE = "<|user|>\n{instruction}\n<|assistant|>\n"


@dataclass(frozen=True)
class Lineage:
    """One alignment chain. `stages` order IS the developmental order."""
    name: str
    checkpoints: tuple[tuple[str, str], ...]     # (stage_name, hf_model_id), in order
    template: str
    refusal_token_piece: str
    # None => NOT YET MEASURED on this family. Run diagnose_refusal_token.py; the drivers
    # refuse to proceed rather than silently score whatever the piece happens to resolve to.
    expected_refusal_id: int | None
    # None => NOT YET VERIFIED. verify_setup.py reports the safe pinned value per lineage;
    # it must be <= the shortest tokenizer-derived eoi_len across the lineage's checkpoints.
    n_eoi: int | None
    notes: str = ""

    @property
    def verified(self) -> bool:
        return self.expected_refusal_id is not None and self.n_eoi is not None


LINEAGES: dict[str, Lineage] = {
    # ---------------------------------------------------------------- VERIFIED
    "zephyr": Lineage(
        name="zephyr",
        checkpoints=(
            ("base", "mistralai/Mistral-7B-v0.1"),
            ("sft",  "alignment-handbook/zephyr-7b-sft-full"),
            ("dpo",  "HuggingFaceH4/zephyr-7b-beta"),
        ),
        template=ZEPHYR_TEMPLATE,
        # MEASURED 2026-09-11 (diagnose_refusal_token.py, zephyr-7b-beta, harmful prompts):
        #   id=28737 piece 'I'  p=0.3675   <- rank 1, what the model ACTUALLY emits
        #   id=315   piece '_I' p=0.000175 <- rank 60
        # Harmless control p(28737)=0.0027 -> a 135x contrast. Both pieces DECODE to "I",
        # which is what hid the error. Resolve with convert_tokens_to_ids, never encode():
        # encode("I") returns [315] because SentencePiece prepends a dummy prefix space.
        refusal_token_piece="I",
        expected_refusal_id=28737,
        # PINNED. Tokenizer-derived eoi_len is 9 (base) vs 10 (SFT/DPO) — the Zephyr
        # tokenizers insert a phantom '' token. A stage-varying window would have silently
        # invalidated the whole cross-stage comparison.
        n_eoi=5,
        notes="ADVERSARIAL CASE: zephyr-7b-beta's DPO deliberately removed safety filtering. "
              "Weak as a sole witness, strong in company — the same coupling signature in a "
              "lineage whose DPO dropped safety data shows the effect belongs to the training "
              "procedure, not the safety dataset.",
    ),
    # ------------------------------------------------------- UNVERIFIED (will not run)
    "olmo2": Lineage(
        name="olmo2",
        checkpoints=(
            ("base", "allenai/OLMo-2-1124-7B"),
            ("sft",  "allenai/OLMo-2-1124-7B-SFT"),
            ("dpo",  "allenai/OLMo-2-1124-7B-DPO"),
            ("rlvr", "allenai/OLMo-2-1124-7B-Instruct"),
        ),
        template=OLMO2_TEMPLATE,
        refusal_token_piece="I",
        expected_refusal_id=None,   # run diagnose_refusal_token.py --lineage olmo2
        n_eoi=None,                 # run verify_setup.py --lineage olmo2
        notes="PRIMARY cross-lineage target: the only fully public 4-point pipeline "
              "(base -> SFT -> DPO -> RLVR) with genuine safety training. ⚠️ model ids, "
              "template and refusal token all UNVERIFIED.",
    ),
    "tulu2": Lineage(
        name="tulu2",
        checkpoints=(
            ("base", "meta-llama/Llama-2-7b-hf"),
            ("sft",  "allenai/tulu-2-7b"),
            ("dpo",  "allenai/tulu-2-dpo-7b"),
        ),
        template=TULU_TEMPLATE,
        refusal_token_piece="I",
        expected_refusal_id=None,
        n_eoi=None,
        notes="Second family (Llama-2), clean SFT/DPO split. ⚠️ UNVERIFIED. Llama-2-7b-hf "
              "is gated on HF — accept the licence first.",
    ),
}


@dataclass(frozen=True)
class Config:
    """A lineage plus the knobs that are held constant across lineages."""
    lineage: str = "zephyr"
    checkpoints: tuple[tuple[str, str], ...] = ()
    template: str = ""
    refusal_token_piece: str = "I"
    expected_refusal_id: int | None = None
    n_eoi: int | None = None

    dtype: str = "bfloat16"
    n_train: int = 128       # samples for the mean-diff direction (Arditi default)
    n_val: int = 32          # samples for the per-layer causal sweep
    prune_layer_pct: float = 0.20   # O-40: last 20% of layers excluded from l* selection
    # Arditi's other two selection criteria (values verified in E01, which reproduced his
    # (pos=-5, layer=12) exactly). kl_threshold is load-bearing: without it, l* selection is
    # an argmax over "how much does ablating this destroy refusal", which happily picks the
    # direction that destroys the MODEL. O-50: this bound gates ABLATION only — applying it
    # to ADDITION reported "no induction" everywhere, including the cell that had to succeed.
    kl_threshold: float = 0.1
    induce_threshold: float = 0.0
    n_control: int = 3       # norm-matched random directions for the negative control
    gen_max_new_tokens: int = 48
    n_behavioral: int = 0    # 0 = the whole held-out tail harmful_train[n_train:]
    n_sample_completions: int | None = None   # None = store every completion (O-44, O-51)
    n_null: int = 16         # shuffled-label directions for the cosine null (see O-49)
    # O-52: n=32 flipped a cell when raised to 256. Size n against the EFFECT being compared
    # (base real 1.4 vs random 1.1), not against convenience.
    n_transplant: int = 256
    batch_size: int = 16
    seed: int = 42
    results_dir: str = "results"
    figures_dir: str = "results/figures"

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(s for s, _ in self.checkpoints)

    def path(self, stage: str, axis: str) -> str:
        """results/{lineage}_{stage}_{axis}.npz — lineage-scoped so families never collide."""
        return f"{self.results_dir}/{self.lineage}_{stage}_{axis}.npz"

    def require_verified(self) -> None:
        """Refuse to run an unverified lineage. O-42 made structural."""
        lin = LINEAGES[self.lineage]
        if lin.verified:
            return
        missing = [n for n, v in (("expected_refusal_id", lin.expected_refusal_id),
                                  ("n_eoi", lin.n_eoi)) if v is None]
        raise SystemExit(
            f"lineage '{self.lineage}' is UNVERIFIED (missing: {', '.join(missing)}).\n"
            f"  {lin.notes}\n\n"
            f"  The refusal token and the end-of-instruction window do NOT transfer between\n"
            f"  model families. Scoring the wrong token produces a clean-looking trend that\n"
            f"  is pure noise (O-42). Before running anything on this lineage:\n"
            f"    1. python diagnose_refusal_token.py --lineage {self.lineage} --stage "
            f"{self.stages[-1]}\n"
            f"       -> read the top-1 token id, set Lineage.expected_refusal_id in config.py\n"
            f"    2. python verify_setup.py --lineage {self.lineage}\n"
            f"       -> it reports the safe pinned n_eoi; set Lineage.n_eoi in config.py\n"
            f"    3. confirm the chat template against the checkpoint's own tokenizer_config")


def config_for(name: str, **overrides) -> Config:
    if name not in LINEAGES:
        raise SystemExit(f"unknown lineage {name!r}. known: {sorted(LINEAGES)}")
    lin = LINEAGES[name]
    return replace(Config(
        lineage=lin.name, checkpoints=lin.checkpoints, template=lin.template,
        refusal_token_piece=lin.refusal_token_piece,
        expected_refusal_id=lin.expected_refusal_id, n_eoi=lin.n_eoi), **overrides)


DEFAULT = config_for("zephyr")
