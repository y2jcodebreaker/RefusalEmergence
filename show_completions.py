"""Print the stored sample completions so a refusal RATE can be sanity-checked as TEXT.

    python show_completions.py            # all stages
    python show_completions.py --stage base

A rate is only as good as its judge. base scoring 0.656 while the single-token axis says
0.113 means at least one of them is not measuring refusal — and the only way to tell which
is to read what the model actually wrote. Each line is tagged with the substring verdict
and, when it fires, the substring responsible, so false positives are visible immediately.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from config import DEFAULT
from refusal_substring import REFUSAL_SUBSTRINGS, is_refusal


def which_substring(text: str) -> str | None:
    low = text.lower()
    for s in REFUSAL_SUBSTRINGS:
        if s.lower() in low:
            return s
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="only this stage")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{DEFAULT.results_dir}/*_refusal.npz"))
    if not paths:
        raise SystemExit("no results — run run_stage.py --behavioral first")

    for p in paths:
        d = np.load(p, allow_pickle=True)
        stage = str(d["stage"])
        if args.stage and stage != args.stage:
            continue
        if "sample_completions" not in d:
            print(f"\n[{stage}] no stored completions (run with --behavioral)")
            continue

        samples = json.loads(str(d["sample_completions"]))
        b = float(d["substring_baseline_rate"]) if "substring_baseline_rate" in d else float("nan")
        a = float(d["substring_ablated_rate"]) if "substring_ablated_rate" in d else float("nan")
        print(f"\n{'=' * 78}\n[{stage}]  substring rate: baseline={b:.3f} -> ablated={a:.3f}\n{'=' * 78}")
        for kind in ("baseline", "ablated"):
            print(f"\n-- {kind} --")
            for i, c in enumerate(samples.get(kind, [])):
                hit = which_substring(c)
                verdict = f"REFUSAL (matched {hit!r})" if is_refusal(c) else "comply"
                print(f"  [{i}] {verdict}\n      {c.strip()[:300]!r}")


if __name__ == "__main__":
    main()
