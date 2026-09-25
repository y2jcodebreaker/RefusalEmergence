"""Re-render results/RUNLOG.md from results/runs.jsonl.

    python rebuild_runlog.py

RUNLOG.md is a human-readable view of the ledger and nothing else, so it can always be
rebuilt. It needs rebuilding because pod_pull.sh resets tracked files under results/ and
RUNLOG.md is tracked: every reset dropped whatever the pod had appended since the last push,
while runs.jsonl survived via the EXIT trap. 60 entries had been lost that way by 2026-09-26.

Refuses to shrink the file: if the rebuild would drop an entry the current file has, the
ledger is missing rows and that is a bigger problem than a stale RUNLOG.
"""

from __future__ import annotations

import json
import re

import runlog

LEDGER, MD = "results/runs.jsonl", "results/RUNLOG.md"


def main() -> None:
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    heads = lambda t: set(re.findall(r"^## .*$", t, re.M))
    before = heads(open(MD).read()) if __import__("os").path.exists(MD) else set()
    tmp = MD + ".rebuild"
    for e in rows:
        runlog._append_md(e, tmp)
    built = open(tmp).read()
    missing = before - heads(built)
    if missing:
        __import__("os").remove(tmp)
        raise SystemExit(f"REFUSING to rebuild: {len(missing)} entries in {MD} are NOT in "
                         f"{LEDGER}, so the ledger has lost rows. First: {sorted(missing)[0]}")
    __import__("os").replace(tmp, MD)
    print(f"rebuilt {MD} from {len(rows)} ledger rows "
          f"({len(heads(built)) - len(before):+d} entries vs before)")


if __name__ == "__main__":
    main()
