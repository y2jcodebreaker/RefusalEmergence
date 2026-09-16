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

# For every other lineage the template is DERIVED from the aligned checkpoint's own
# tokenizer chat_template (verify_setup.py prints it), never transcribed by hand. Hand-copied
# special tokens are the same class of error as O-42: wrong, plausible, and silent.


@dataclass(frozen=True)
class Lineage:
    """One alignment chain. `stages` order IS the developmental order."""
    name: str
    checkpoints: tuple[tuple[str, str], ...]     # (stage_name, hf_model_id), in order
    # None => derive it from the ALIGNED checkpoint's tokenizer chat_template and paste it
    # here (verify_setup.py prints it). Applied to every stage, base included: the base model
    # has no template of its own, and holding the prompt format constant is what makes the
    # weights the only variable.
    template: str | None
    refusal_token_piece: str
    # None => NOT YET MEASURED on this family. Run diagnose_refusal_token.py; the drivers
    # refuse to proceed rather than silently score whatever the piece happens to resolve to.
    expected_refusal_id: int | None
    # None => NOT YET VERIFIED. verify_setup.py reports the safe pinned value per lineage;
    # it must be <= the shortest tokenizer-derived eoi_len across the lineage's checkpoints.
    n_eoi: int | None
    # PER-STAGE REGIME OVERRIDE: {stage: (template, refusal_token_id, n_eoi)}.
    #
    # Needed because a checkpoint can be OUTSIDE a usable operating regime under the
    # lineage's chat template. base OLMo 2 degenerates into verbatim prompt echo under
    # '<|user|>...<|assistant|>' -- it produces no judge-able text at all -- but under a
    # plain 'User:/Assistant:' format it is fully fluent and COMPLIES with harmful requests
    # (O-56/O-57). A model that cannot be prompted cannot be asked whether it can be steered.
    #
    # The refusal token changes WITH the template, which is the O-42 trap inverted: after
    # '<|assistant|>\n' the model emits bare 'I' (40); after 'Assistant:' it emits
    # space-prefixed ' I' (358), and 40 there has p ~= 1e-6. So expected_refusal_id is a
    # function of (family, TEMPLATE), not family alone.
    #
    # Cost, stated not hidden: a stage on an override is NOT prompt-format-matched to the
    # rest of its lineage, so raw scores are not comparable across that boundary. What
    # remains comparable is the WITHIN-STAGE question -- "can this model be steered into
    # refusing at all?" -- which is the question the coupling claim actually rests on.
    stage_regime: dict[str, tuple[str, int, int]] | None = None
    notes: str = ""

    @property
    def verified(self) -> bool:
        return (self.expected_refusal_id is not None and self.n_eoi is not None
                and self.template is not None)


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
    # PRIMARY cross-lineage target. Ids CONFIRMED against the HF API 2026-09-14 (all four
    # resolve; base is Olmo2ForCausalLM, 84k downloads, pretrained-only tags).
    "olmo2": Lineage(
        name="olmo2",
        checkpoints=(
            ("base", "allenai/OLMo-2-1124-7B"),
            ("sft",  "allenai/OLMo-2-1124-7B-SFT"),
            ("dpo",  "allenai/OLMo-2-1124-7B-DPO"),
            ("rlvr", "allenai/OLMo-2-1124-7B-Instruct"),
        ),
        # DERIVED 2026-09-14 from allenai/OLMo-2-1124-7B-Instruct's own chat_template, not
        # hand-written. The raw render begins with the BOS '<|endoftext|>', which the
        # tokenizer re-adds at encode time -- stripped here so it does not appear twice.
        template="<|user|>\n{instruction}\n<|assistant|>\n",
        refusal_token_piece="I",
        # MEASURED 2026-09-16 (diagnose_refusal_token.py, OLMo-2-1124-7B-Instruct, harmful):
        #   id=40  piece 'I'   p=0.9483  <- rank 0
        #   id=358 piece ' I'  p=0.000008 <- rank 65
        # Harmless control p(40)=0.0046 -> a 205x contrast (Zephyr's was 135x). The bare-vs-
        # space-prefixed trap recurs on a completely different tokenizer, with different ids.
        expected_refusal_id=40,
        # n_eoi=5, NOT the full derived 6. At 6 the window's first token is BPE-merged with
        # the instruction's final character -- '?\n' becomes one token '?Ċ' -- so the window
        # varies with the prompt and a layer-0 probe can read "does this end with a question
        # mark?". Harmless prompts (MMLU questions) end in '?' far more often than harmful
        # ones (imperatives), which is why the probe's L0 read 0.644 instead of chance.
        # At 5 the window is ['<','|','assistant','|','>Ċ'], constant across all prompts.
        # Zephyr escapes this because its suffix begins with '</s>', a special token that
        # never merges. verify_setup now checks this automatically (O-58).
        n_eoi=5,
        # base is evaluated under a PLAIN template, in its own valid regime (O-56/O-57).
        # Verified 2026-09-16: fluent, on-task, and complies with all three sampled harmful
        # prompts, so its 0.000 refusal rate is real rather than an artefact of echoing.
        # Token 358 (' I', rank 5) not 40 ('I', rank >2000) -- the token follows the template.
        # n_eoi=2, not 3: same BPE-merge leak — at 3 the leading 'Ċ' absorbs the
        # instruction's final character. ['Assistant',':'] is constant across all prompts.
        stage_regime={"base": ("User: {instruction}\nAssistant:", 358, 2)},
        notes="The fully public 4-point pipeline (base -> SFT -> DPO -> RLVR) with GENUINE "
              "safety training — post-trained on an OLMo variant of Tulu 3. Preferred over "
              "Olmo 3 as the first cross-lineage run because Olmo2ForCausalLM has been "
              "supported in transformers far longer, so it is the lower-risk replication.",
    ),
    # Newer, same 4-point structure. Ids CONFIRMED 2026-09-14. Note the capitalisation change
    # (OLMo -> Olmo) and that the base is date-stamped while the rest are not: guessing any of
    # these would have 404'd.
    "olmo3": Lineage(
        name="olmo3",
        checkpoints=(
            ("base", "allenai/Olmo-3-1025-7B"),
            ("sft",  "allenai/Olmo-3-7B-Instruct-SFT"),
            ("dpo",  "allenai/Olmo-3-7B-Instruct-DPO"),
            ("rlvr", "allenai/Olmo-3-7B-Instruct"),
        ),
        # DERIVED 2026-09-14, with the default system turn REMOVED. Olmo-3-7B-Instruct's
        # own template injects "You are a helpful function-calling AI assistant..." — text
        # about function calling is irrelevant to refusal, and the base model has no system
        # prompt at all, so including it would add a variable the comparison does not want.
        # diagnose_refusal_token.py prints a with-system-turn panel; check both before
        # committing to this choice.
        template="<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        refusal_token_piece="I",
        expected_refusal_id=None,
        n_eoi=None,
        notes="Olmo3ForCausalLM — needs a transformers new enough to know the architecture. "
              "Uses a ChatML-style template (<|im_start|>/<|im_end|>), unlike OLMo 2's "
              "<|user|>/<|assistant|>, so the derived template will differ. More current for "
              "a 2027 submission; run it after olmo2 succeeds.",
    ),
    "tulu2": Lineage(
        name="tulu2",
        checkpoints=(
            ("base", "meta-llama/Llama-2-7b-hf"),
            ("sft",  "allenai/tulu-2-7b"),
            ("dpo",  "allenai/tulu-2-dpo-7b"),
        ),
        template=None,
        refusal_token_piece="I",
        expected_refusal_id=None,
        n_eoi=None,
        notes="Third family (Llama-2), clean SFT/DPO split. ⚠️ ids NOT yet confirmed against "
              "the HF API. Llama-2-7b-hf is gated — accept the licence first.",
    ),
}


@dataclass(frozen=True)
class Config:
    """A lineage plus the knobs that are held constant across lineages."""
    lineage: str = "zephyr"
    checkpoints: tuple[tuple[str, str], ...] = ()
    template: str | None = None
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

    def regime(self, stage: str) -> tuple[str, int, int, bool]:
        """(template, refusal_token_id, n_eoi, is_override) for one stage.

        Returns the lineage defaults unless that stage has an explicit override, in which
        case the caller MUST log that this stage is not format-matched to the others."""
        ov = (LINEAGES[self.lineage].stage_regime or {}).get(stage)
        if ov is None:
            return self.template, self.expected_refusal_id, self.n_eoi, False
        return ov[0], ov[1], ov[2], True

    def path(self, stage: str, axis: str) -> str:
        """results/{lineage}_{stage}_{axis}.npz — lineage-scoped so families never collide."""
        return f"{self.results_dir}/{self.lineage}_{stage}_{axis}.npz"

    def require_verified(self) -> None:
        """Refuse to run an unverified lineage. O-42 made structural."""
        lin = LINEAGES[self.lineage]
        if lin.verified:
            return
        missing = [n for n, v in (("template", lin.template),
                                  ("expected_refusal_id", lin.expected_refusal_id),
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
            f"    3. paste the template verify_setup.py derives from the aligned\n"
            f"       checkpoint's own tokenizer chat_template — do NOT hand-write it")


def config_for(name: str, **overrides) -> Config:
    if name not in LINEAGES:
        raise SystemExit(f"unknown lineage {name!r}. known: {sorted(LINEAGES)}")
    lin = LINEAGES[name]
    return replace(Config(
        lineage=lin.name, checkpoints=lin.checkpoints, template=lin.template,
        refusal_token_piece=lin.refusal_token_piece,
        expected_refusal_id=lin.expected_refusal_id, n_eoi=lin.n_eoi), **overrides)


DEFAULT = config_for("zephyr")
