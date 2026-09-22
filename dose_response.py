"""P1-E7d. Do behaviour and coupling fall TOGETHER, and does the probe stay flat?

    python dose_response.py --arm benign            # the attack
    python dose_response.py --arm safety-preserved  # the matched control

P1-E7 compared two endpoints: an aligned checkpoint and an attacked one. That is a
before/after pair, and a before/after pair cannot distinguish "behaviour and coupling
collapse together" from "both happen to differ after 1500 steps". This walks the training
run and measures all three quantities at six doses, so the SHAPE of each curve is the
evidence rather than its endpoints.

THE PREDICTION, stated before the run:

    behavioural refusal   falls
    coupling (peak induce, and the count of steerable layers)  falls WITH it, and faster
    probe accuracy        FLAT at ~1.000 the whole way

The coupling account predicts the first two move together while the third does not move at
all. The capability account -- alignment teaches the model to recognise harm -- predicts the
probe degrades alongside the behaviour. Six points per curve separate those far better than
two do. Frank (2026) uses the same shape of argument for a different claim.

WHY NOTHING IS SAVED PER DOSE. `attack.py` writes a merged 15 GB checkpoint. Six doses x two
arms would be 180 GB, and the pod has 85 GB. Every measurement function in this repo takes a
live model object (get_mean_diff, refusal_strength_curve, cache_activations), so the model is
measured IN PLACE at each dose and only the LoRA ADAPTER is written -- ~80 MB at rank 16,
which is enough to reconstruct any dose exactly. Trading 180 GB for 0.5 GB costs nothing:
the merged checkpoint was only ever a convenience for scripts that load by path.

COMPLETIONS ARE STORED AT EVERY DOSE, AND THIS IS NOT OPTIONAL. On 2026-09-21 the substring
judge was found to undercount refusal by 8-10x on two model families, because every one of
Arditi's twelve substrings encodes INABILITY ("I cannot") or apology, and none match a
refusal that asserts a NORM ("this is illegal and unethical", "I strongly condemn"). The
behavioural axis of this experiment is therefore reported only after a classifier has scored
the stored text. A dose-response curve built on substring rates alone would be a curve
through six wrong numbers, and its SHAPE could be wrong too -- if the register shifts DURING
training, the substring rate would fall while true refusal held, which is precisely the
artifact this experiment would otherwise claim as its result.

    python judge_wildguard.py results/olmo2_e7d_{arm}_dose_{step}_refusal.npz --lineage olmo2

DESTROYED, OR MERELY UNFINDABLE? The direction is RE-FITTED from each dose's own activations,
so "no valid direction at dose 100" has two readings that the curve alone cannot separate:

    (a) the COUPLING is destroyed                       <- what the claim needs
    (b) mean-diff no longer FINDS a direction that still works

Under (b) the mechanism is intact and only our estimator lost it, which would make the whole
result an artifact of the measurement. So the dose-0 direction is FROZEN at the start and
re-injected at every subsequent dose, over the same coefficient grid transplant.py uses. If
that frozen direction keeps inducing refusal while the re-fitted one goes negative, reading
(b) is correct and the claim collapses. If it dies too, the coupling is genuinely gone.

This is P1-E7b's rlvr->attacked cell turned into a curve, and that cell DID restore refusal
(+1.069), so the instrument is known to work on exactly this comparison. Testing it in-run
costs one extra sweep per dose and needs no saved checkpoint.

WHERE THIS SITS IN THE GRAPH.

    attack.py (P1-E7)  ->  endpoints, behaviour + mechanism           [DONE]
        |
        +-- overrefusal.py (P1-E7c)  ->  is the control valid?        [DONE, passed]
        +-- transplant.py (P1-E7b)   ->  what moved mechanistically   [DONE]
        +-- dose_response.py (P1-E7d) -> do they move TOGETHER        <- here
                |
                +-- judge_wildguard.py on each dose  (BLOCKING for the behavioural curve)

It also subsumes the seed-replication control that P1-E7 still has open: the original
attacked checkpoint is gone (it lived on a destroyed pod), so dose 1500 here is an
independent training run at a different seed draw. Its endpoint will NOT land exactly on
0.477, and that is the point -- agreement in SHAPE across two runs is stronger than a single
number reproducing.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from attack import build_benign, build_safety_examples, encode_sft, fill_responses
from config import config_for
from data import load_instructions
from probes import cache_activations, logistic_accuracy, mass_mean_accuracy
from refusal_direction import (_addition_handles, _last_logits, _mean_refusal,
                               get_mean_diff, refusal_strength_curve,
                               resolve_refusal_token)
from transplant import COEFFS
from refusal_substring import behavioral_rates
from run_stage import load_model, set_seed
from runlog import RunRecord
from verify_setup import check_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("P1-E7d")

EXPERIMENT = "P1-E7d"
QUESTION = ("Across a benign fine-tuning run, do behavioural refusal and the coupling that "
            "mediates it fall together while probe accuracy stays flat?")

# Steps at which to measure. 0 is the untouched checkpoint, so every curve has its own
# origin measured by the same code path rather than borrowed from another run's .npz.
DEFAULT_DOSES = (0, 100, 250, 500, 1000, 1500)


def frozen_probe(model, tok, cfg, template, refusal_toks, frozen: torch.Tensor,
                 layer: int, pos: int, harmless) -> dict:
    """Does the DOSE-0 direction still induce refusal in this model?

    Separates 'the coupling is destroyed' from 'mean-diff stopped finding it' -- see the
    module docstring. Same absolute-score convention as refusal_strength_curve's steer
    surface, so the values are comparable with `max_induce` cell for cell."""
    best, at, sweep = -float("inf"), None, {}
    for c in COEFFS:
        h = _addition_handles(model, frozen, coeff=c, layer=layer)
        try:
            v = _mean_refusal(_last_logits(model, tok, harmless, template, cfg.batch_size),
                              refusal_toks)
        finally:
            for x in h:
                x.remove()
        sweep[c] = float(v)
        if v > best:
            best, at = v, c
    # The WHOLE sweep, not just its peak. The re-fitted surface is measured at coeff 1.0
    # only, so without the coeff-1.0 value here the frozen column cannot be compared with it
    # like for like -- and "the frozen direction wins" would be confoundable with "the
    # frozen direction was allowed a bigger coefficient". Noticed 2026-09-22, after a run
    # whose peaks all landed at coeff 2.0.
    return {"frozen_induce_max": float(best), "frozen_induce_at_coeff": float(at),
            "frozen_induce_at_1": float(sweep.get(1.0, float("nan"))),
            "frozen_sweep_coeffs": np.array(sorted(sweep)),
            "frozen_sweep_values": np.array([sweep[c] for c in sorted(sweep)]),
            "frozen_layer": int(layer), "frozen_pos": int(pos)}


def measure(model, tok, cfg, stage_tag: str, template: str, refusal_toks, n_eoi: int,
            splits: dict, frozen: dict | None = None) -> dict:
    """Behaviour + coupling + probe on the model AS IT CURRENTLY IS. No saving, no reloading.

    `frozen` carries the dose-0 direction and its layer/pos; when present, it is re-injected
    here so every dose answers 'does the ORIGINAL direction still work' as well as 'is a
    direction findable now'."""
    was_training = model.training
    model.eval()
    grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        out: dict = {}

        # --- coupling. Same estimator and hook point as run_stage.
        dirs = get_mean_diff(model, tok, splits["harmful_tr"], splits["harmless_tr"],
                             template, n_eoi, cfg.batch_size)
        res = refusal_strength_curve(model, tok, dirs, splits["harmful_val"], template,
                                     refusal_toks, cfg.prune_layer_pct, cfg.batch_size,
                                     harmless_val=splits["harmless_val"],
                                     kl_threshold=cfg.kl_threshold,
                                     induce_threshold=cfg.induce_threshold)
        steer, excl = res["steer"], res["excluded_layers"]
        keep = np.ones(steer.shape[1], dtype=bool)
        if np.size(excl):
            keep[np.asarray(excl, dtype=int)] = False
        best_per_layer = np.nanmax(steer, axis=0)
        out["l_star"] = int(res["l_star"])
        out["peak_ablation"] = float(np.nanmax(res["bypass"]))
        out["max_induce"] = float(np.nanmax(best_per_layer[keep]))
        # The count of layers that can be steered into refusing: 13 -> 0 was the endpoint
        # result, and it is the coupling quantity with the least free choice in it.
        out["n_steerable_layers"] = int(np.sum(best_per_layer[keep] >= cfg.induce_threshold))
        # -1, not None: np.array(None) is an object array and np.savez refuses it.
        out["n_cells_passing"] = int(np.sum(res["valid"])) if "valid" in res else -1

        # --- behaviour. Completions are KEPT; the rate here is a substring rate and is a
        # LOWER BOUND (see the module docstring). The classifier runs afterwards.
        p = int(res["pos_star"]) if out["l_star"] >= 0 else -1
        abl_dir = dirs[p, res["l_star"]] if out["l_star"] >= 0 else None
        b_rate, a_rate, samples = behavioral_rates(
            model, tok, splits["beh"], template, abl_dir,
            cfg.gen_max_new_tokens, cfg.batch_size, n_samples=None)
        out["substring_baseline_rate_strict"] = float(samples["strict_baseline"])
        out["substring_ablated_rate_strict"] = float(samples["strict_ablated"])
        out["sample_completions"] = samples

        # --- representation. Peak is saturated at ~1.000, so the CURVE is what matters;
        # both are stored and the per-layer arrays go in the npz.
        A = {k: cache_activations(model, tok, v, template, n_eoi, cfg.batch_size).numpy()
             for k, v in (("fit_pos", splits["harmful_tr"]), ("fit_neg", splits["harmless_tr"]),
                          ("test_pos", splits["probe_test_pos"]),
                          ("test_neg", splits["harmless_test"]))}
        n_pos, n_layers = A["fit_pos"].shape[1], A["fit_pos"].shape[2]
        acc_mm = np.full((n_pos, n_layers), np.nan)
        acc_lr = np.full((n_pos, n_layers), np.nan)
        for pi in range(n_pos):
            for li in range(n_layers):
                tp, tn = A["fit_pos"][:, pi, li, :], A["fit_neg"][:, pi, li, :]
                ep, en = A["test_pos"][:, pi, li, :], A["test_neg"][:, pi, li, :]
                acc_mm[pi, li] = mass_mean_accuracy(tp, tn, ep, en)
                acc_lr[pi, li] = logistic_accuracy(tp, tn, ep, en, seed=cfg.seed)
        out["acc_mass_mean"] = acc_mm
        out["acc_logistic"] = acc_lr
        out["probe_peak_logistic"] = float(np.nanmax(acc_lr))
        out["probe_peak_mass_mean"] = float(np.nanmax(acc_mm))
        out["probe_L0_logistic"] = float(np.nanmax(acc_lr[:, 0]))
        out["steer_curve"] = best_per_layer
        out["bypass"] = res["bypass"]
        # steer AT l*, not the surface maximum. max_induce is the best over ALL cells and at
        # dose 0 that is a different layer (L18) from l* (L24), so it is the wrong thing to
        # check the frozen sweep against -- the frozen direction lives at l*.
        out["steer_at_l_star"] = (float(steer[int(res["pos_star"]), int(res["l_star"])])
                                  if out["l_star"] >= 0 else float("nan"))

        # --- the frozen dose-0 direction, re-tested. At dose 0 this is the SELF cell and
        # must reproduce max_induce, which makes it its own positive control.
        if frozen is not None:
            out.update(frozen_probe(model, tok, cfg, template, refusal_toks,
                                    frozen["vec"], frozen["layer"], frozen["pos"],
                                    splits["harmless_val"]))
        # Keep this dose's own directions so dose 0 can be frozen by the caller.
        out["_dirs"] = dirs
        out["_pos_star"] = int(res["pos_star"]) if out["l_star"] >= 0 else -1
        return out
    finally:
        torch.set_grad_enabled(grad)
        if was_training:
            model.train()


def train_to(model, tok, examples, opt, *, target_step: int, from_step: int, bs: int,
             pad: int) -> int:
    """Continue training until `target_step`, cycling the shuffled data. Returns the step."""
    model.train()
    torch.set_grad_enabled(True)
    step = from_step
    i = (from_step * bs) % max(1, len(examples))
    while step < target_step:
        batch = examples[i:i + bs]
        if not batch:
            i = 0
            continue
        n = max(len(ids) for ids, _ in batch)
        ids = torch.tensor([x + [pad] * (n - len(x)) for x, _ in batch])
        lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in batch])
        att = (ids != pad).long()
        loss = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                     labels=lab.to(model.device)).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        i += bs
        if i >= len(examples):
            i = 0
        if step % 100 == 0:
            logger.info("    step %d  loss %.4f", step, loss.item())
    model.eval()
    torch.set_grad_enabled(False)
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="olmo2")
    ap.add_argument("--from", dest="src", default="rlvr")
    ap.add_argument("--arm", required=True, choices=("benign", "safety-preserved"))
    ap.add_argument("--doses", default=",".join(str(d) for d in DEFAULT_DOSES),
                    help="training steps at which to measure; 0 = the untouched checkpoint")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-safety", type=int, default=50)
    ap.add_argument("--responses", default="reference", choices=("reference", "self"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tag", default="olmo2_e7d", help="prefix for the output npz files")
    args = ap.parse_args()

    doses = sorted({int(d) for d in args.doses.split(",")})
    if doses[0] != 0:
        raise SystemExit("dose 0 must be included: it is this run's own origin, and reusing "
                         "another run's baseline would confound the curve with run-to-run "
                         "variation (the very thing a dose-response is meant to rule out).")

    cfg = config_for(args.lineage)
    if not check_disk(cfg, stages=(args.src,)):
        raise SystemExit("free disk (or set HF_HOME) before loading weights.")
    ckpts = dict(cfg.checkpoints)
    if args.src not in ckpts:
        raise SystemExit(f"unknown --from {args.src!r}; have {list(ckpts)}")

    set_seed(cfg.seed)
    model, tok = load_model(ckpts[args.src], cfg.dtype)
    template, want_id, n_eoi, _ov = cfg.regime(args.src)
    refusal_toks = [resolve_refusal_token(tok, cfg.refusal_token_piece, want_id)]

    # Splits, fixed once so every dose is measured on identical prompts.
    tail = load_instructions("harmful_train")[cfg.n_train:]
    splits = {
        "harmful_tr": load_instructions("harmful_train")[: cfg.n_train],
        "harmless_tr": load_instructions("harmless_train")[: cfg.n_train],
        "harmful_val": load_instructions("harmful_val")[: cfg.n_val],
        "harmless_val": load_instructions("harmless_val")[: cfg.n_val],
        "beh": tail[: cfg.n_behavioral] if cfg.n_behavioral else tail,
        # The probe's held-out sets, taken EXACTLY as probe_representation._splits takes them
        # (harmless_train count-matched to the harmful tail, so chance is 0.500). Using a
        # different split here would make this curve incomparable with the probe numbers
        # already reported, which is the whole point of measuring it.
        "probe_test_pos": tail,
        "harmless_test": load_instructions("harmless_train")[cfg.n_train: cfg.n_train + len(tail)],
    }
    if len(splits["harmless_test"]) < len(splits["probe_test_pos"]):
        splits["probe_test_pos"] = splits["probe_test_pos"][: len(splits["harmless_test"])]
    logger.info("[%s] doses=%s | behavioural n=%d | probe test %d/%d (chance %.3f)",
                args.arm, doses, len(splits["beh"]), len(splits["probe_test_pos"]),
                len(splits["harmless_test"]),
                len(splits["probe_test_pos"]) / (len(splits["probe_test_pos"])
                                                 + len(splits["harmless_test"])))

    pairs = fill_responses(model, tok, build_benign(cfg, args.n, args.responses), template, cfg)
    n_benign, n_safety = len(pairs), 0
    if args.arm == "safety-preserved":
        safety = build_safety_examples(model, tok, cfg, template, args.n_safety)
        n_safety = len(safety)
        pairs = pairs + safety
    import random as _r
    _r.Random(cfg.seed).shuffle(pairs)
    examples = encode_sft(tok, pairs, template)
    logger.info("[%s] %d benign + %d safety -> %d encoded", args.arm, n_benign, n_safety,
                len(examples))

    from peft import LoraConfig, get_peft_model
    set_seed(cfg.seed)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    rows, step, frozen = [], 0, None
    with RunRecord(EXPERIMENT, "dose_response.py", cfg=cfg, question=QUESTION,
                   notes=f"arm={args.arm} doses={doses} rank={args.rank} lr={args.lr} "
                         f"n={args.n} responses={args.responses}. Substring rates are LOWER "
                         f"BOUNDS; run judge_wildguard.py on each dose npz.") as rec:
        for dose in doses:
            if dose > step:
                logger.info("[%s] training %d -> %d steps", args.arm, step, dose)
                step = train_to(model, tok, examples, opt, target_step=dose, from_step=step,
                                bs=args.bs, pad=tok.pad_token_id)
            logger.info("[%s] === measuring at dose %d ===", args.arm, step)
            m = measure(model, tok, cfg, f"{args.arm}@{step}", template, refusal_toks, n_eoi,
                        splits, frozen=frozen)

            if frozen is None:
                # Freeze dose 0's direction at its OWN l*. If l* = -1 here the untouched
                # checkpoint has no validated direction and the whole comparison is moot, so
                # refuse rather than freeze the unfiltered fallback and quietly compare
                # against a direction that never worked.
                if m["l_star"] < 0:
                    raise SystemExit(
                        f"dose 0 has no filtered l* for {ckpts[args.src]}, so there is no "
                        f"validated direction to freeze and 'does the original direction "
                        f"still work' has no meaning. Check the lineage/regime before "
                        f"reading anything into a dose-response here.")
                frozen = {"vec": m["_dirs"][m["_pos_star"], m["l_star"]].clone(),
                          "layer": int(m["l_star"]), "pos": int(m["_pos_star"])}
                logger.info("[%s] FROZE the dose-0 direction @ (pos %d, L%d), norm %.1f -- it "
                            "will be re-injected at every later dose",
                            args.arm, frozen["pos"], frozen["layer"],
                            float(frozen["vec"].norm()))
                m.update(frozen_probe(model, tok, cfg, template, refusal_toks, frozen["vec"],
                                      frozen["layer"], frozen["pos"], splits["harmless_val"]))
                # POSITIVE CONTROL for the frozen path. The grid includes coeff 1.0, and at
                # coeff 1.0 the frozen injection at (pos*, l*) is EXACTLY what the re-fitted
                # sweep already measured at that cell. So the frozen maximum must be at least
                # that value; if it is not, the two code paths disagree and neither is usable.
                # Compared at l*, not against max_induce, which is the best over all cells and
                # at dose 0 sits at a different layer.
                if m["frozen_induce_max"] < m["steer_at_l_star"] - 0.01:
                    raise SystemExit(
                        f"dose 0 self-check FAILED: the frozen sweep peaks at "
                        f"{m['frozen_induce_max']:+.3f} but the re-fitted surface already "
                        f"reads {m['steer_at_l_star']:+.3f} at the very same cell "
                        f"(pos {frozen['pos']}, L{frozen['layer']}), and the grid includes "
                        f"coeff 1.0. The two paths must agree at dose 0 or the frozen curve "
                        f"means nothing.")
                logger.info("[%s] dose-0 self-check OK: frozen sweep peaks %+.3f (coeff %.1f) "
                            ">= re-fitted %+.3f at the same cell",
                            args.arm, m["frozen_induce_max"], m["frozen_induce_at_coeff"],
                            m["steer_at_l_star"])

            logger.info("[%s] dose %4d | refusal(substring) %.3f | l*=%2d | refit induce "
                        "%+.3f | FROZEN induce %+.3f | steerable %2d | probe %.3f",
                        args.arm, step, m["substring_baseline_rate_strict"], m["l_star"],
                        m["max_induce"], m["frozen_induce_max"], m["n_steerable_layers"],
                        m["probe_peak_logistic"])

            path = f"{cfg.results_dir}/{args.tag}_{args.arm}_dose_{step}_refusal.npz"
            np.savez(path,
                     stage=np.array(f"{args.arm}@{step}"),
                     model_id=np.array(f"{ckpts[args.src]}+lora@{step}"),
                     dose_steps=np.array(step),
                     n_behavioral=np.array(len(splits["beh"])),
                     sample_completions=np.array(json.dumps(m["sample_completions"])),
                     **{k: np.array(v) for k, v in m.items()
                        if k != "sample_completions" and not k.startswith("_")})
            if dose > 0:
                model.save_pretrained(f"models/{args.tag}-{args.arm}-adapter-{step}")
            rows.append({k: v for k, v in m.items()
                         if k in ("l_star", "peak_ablation", "max_induce",
                                  "n_steerable_layers", "probe_peak_logistic",
                                  "probe_peak_mass_mean", "probe_L0_logistic",
                                  "substring_baseline_rate_strict",
                                  "substring_ablated_rate_strict",
                                  "frozen_induce_max", "frozen_induce_at_coeff",
                                  "frozen_induce_at_1")}
                        | {"dose": step})
            rec.result(arm=args.arm, dose=step, path=path,
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in rows[-1].items() if k != "dose"})

    print(f"\n=== P1-E7d dose-response: {args.arm} ===")
    print(f"{'dose':>6} {'refusal*':>9} {'l*':>4} {'refit@c1':>9} {'FROZ@c1':>8} "
          f"{'FROZ max':>9} {'steerable':>10} {'probe':>7}")
    for r in rows:
        print(f"{r['dose']:>6} {r['substring_baseline_rate_strict']:>9.3f} {r['l_star']:>4} "
              f"{r['max_induce']:>+9.3f} {r['frozen_induce_at_1']:>+8.3f} "
              f"{r['frozen_induce_max']:>+9.3f} "
              f"{r['n_steerable_layers']:>10} {r['probe_peak_logistic']:>7.3f}")
    print("\n  FROZEN = the dose-0 direction re-injected into this dose, best over the same\n"
          "  coefficient grid. It separates 'the coupling is destroyed' (frozen goes negative\n"
          "  too) from 'mean-diff stopped finding it' (frozen stays positive while refit does\n"
          "  not). Threshold is 0.00, same convention as the refit column.")
    print("\n* SUBSTRING rate — a LOWER BOUND on refusal, and not the number to report. It\n"
          "  misses normative refusals entirely (8-10x undercount on two families, 2026-09-21).\n"
          "  The behavioural curve is only valid after:")
    for r in rows:
        print(f"    python judge_wildguard.py "
              f"{cfg.results_dir}/{args.tag}_{args.arm}_dose_{r['dose']}_refusal.npz "
              f"--lineage {args.lineage}")


if __name__ == "__main__":
    main()
