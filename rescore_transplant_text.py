"""Re-score a saved P1-E1c completions file with the CURRENT judge. No GPU, no model.

    python rescore_transplant_text.py results/olmo2_base_from_sft_text.json

transplant_text.py stores every completion, not just the rates, so when the judge changes the
old runs do not have to be repeated on a pod -- they are re-read. The first version of the
degenerate judge missed phrase-level loops and scored a looping arm at 0.000; this recovers
the corrected numbers from the run that is already on disk.
"""

from __future__ import annotations

import argparse
import json

from refusal_substring import refusal_rate
from transplant_text import degenerate_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="results/{lineage}_{target}_from_{source}_text.json")
    ap.add_argument("--show", type=int, default=0, help="print this many completions per arm")
    args = ap.parse_args()
    with open(args.path) as f:
        data = json.load(f)

    print(f"{args.path}\n{'arm':<12}{'coeff':>8}{'refusal':>10}{'strict':>9}{'degenerate':>13}")
    for k, v in data.items():
        kind, coeff = k.rsplit("|", 1)
        c = v["completions"]
        print(f"{kind:<12}{float(coeff):>8.1f}{refusal_rate(c):>10.3f}"
              f"{refusal_rate(c, strict=True):>9.3f}{degenerate_rate(c):>13.3f}")
        for s in c[: args.show]:
            print("      | " + s.strip().replace("\n", " ")[:100])


if __name__ == "__main__":
    main()
