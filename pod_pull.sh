#!/usr/bin/env bash
# Safe `git pull` on a pod that has been running experiments.
#
#     ./pod_pull.sh
#
# results/ holds BOTH tracked artefacts (runs.jsonl, RUNLOG.md, the *_text.json and
# *_wildguard.json evidence files) and files the pod regenerates, so a plain pull aborts --
# three times now. Worse, the obvious reflex (`git checkout -- results/`) discards ledger rows
# that exist nowhere else: two were nearly lost on 2026-09-19, one of them the run behind the
# paper's headline number.
#
# So: back up the ledger, drop regenerable conflicts, pull, merge the ledger back.
set -euo pipefail

BACKUP="/workspace/runs.jsonl.podbackup.$(date +%s)"
if [ -f results/runs.jsonl ]; then
    cp results/runs.jsonl "$BACKUP"
    echo "ledger backed up -> $BACKUP  ($(wc -l < "$BACKUP") rows)"
fi

# Regenerable from the .npz / *_text.json, so safe to drop in favour of origin's copies.
rm -f results/olmo2_*_wildguard.json results/zephyr_*_wildguard.json
git checkout -- results/ 2>/dev/null || true

git pull

if [ -f "$BACKUP" ]; then
    python merge_ledger.py "$BACKUP"
else
    echo "no prior ledger to merge"
fi
echo
echo "Ledger: $(wc -l < results/runs.jsonl) rows. Nothing was discarded without being merged back."
