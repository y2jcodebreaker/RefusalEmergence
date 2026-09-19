"""Union two run ledgers without losing a row. Idempotent.

    python merge_ledger.py /workspace/runs.jsonl.podbackup

The ledger is the one artefact here that cannot be regenerated, and it is TRACKED, so a
`git pull` onto a pod that has been running experiments will happily overwrite it. That is not
hypothetical: two rows were nearly lost on 2026-09-19, one of them the run behind the paper's
headline number (z=+3.7, 0/160).

Rows are identified by (started_utc, script), so merging is order-independent and safe to
repeat. Nothing is ever dropped -- on a conflicting duplicate the incoming row is kept and the
difference is reported, because a silent pick is how provenance rots.
"""

from __future__ import annotations

import argparse
import json
import os

LEDGER = "results/runs.jsonl"


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def merge(repo: list[dict], other: list[dict]) -> tuple[list[dict], list[dict], list[tuple]]:
    """(merged, added, conflicts). A conflict is the same (time, script) with different bodies."""
    by_key = {(r["started_utc"], r["script"]): r for r in repo}
    added, conflicts = [], []
    for r in other:
        k = (r["started_utc"], r["script"])
        if k not in by_key:
            by_key[k] = r
            added.append(r)
        elif by_key[k] != r:
            conflicts.append((k, by_key[k], r))
    return sorted(by_key.values(), key=lambda r: r["started_utc"]), added, conflicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("other", help="the other ledger, e.g. a pod backup")
    ap.add_argument("--into", default=LEDGER)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo, other = load(args.into), load(args.other)
    merged, added, conflicts = merge(repo, other)
    print(f"{args.into}: {len(repo)} rows  +  {args.other}: {len(other)} rows "
          f"->  {len(merged)} ({len(added)} recovered)")

    # WRITE FIRST, REPORT SECOND. The report used to come first, and a row missing an
    # optional key crashed the f-string BEFORE the merged file was written -- so a run that
    # had already computed the correct union left the ledger truncated to the repo's copy.
    # Caught 2026-09-19 by a test that fed this a row with no `experiment` field. The durable
    # side effect must not be downstream of anything that can raise, least of all formatting.
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        tmp = args.into + ".tmp"
        with open(tmp, "w") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, args.into)      # atomic: a crash mid-write cannot truncate the ledger
        print(f"wrote {args.into}")

    def _f(v: object, w: int) -> str:
        return f"{v if v is not None else '-':<{w}}"

    for r in added:
        print(f"  RECOVERED {r.get('started_utc', '?')}  {_f(r.get('experiment'), 7)} "
              f"{_f(r.get('script'), 24)} {_f(r.get('status'), 7)} "
              f"@{(r.get('git') or {}).get('short')}")
    for k, a, b in conflicts:
        print(f"  ⚠️  CONFLICT at {k}: same run recorded twice with different bodies. Kept the "
              f"existing one; statuses {a.get('status')} vs {b.get('status')}. Resolve by hand.")


if __name__ == "__main__":
    main()
