#!/usr/bin/env bash
# Safe `git pull` on a pod that has been running experiments.
#
#     ./pod_pull.sh
#
# results/ holds BOTH tracked artefacts (runs.jsonl, RUNLOG.md, the *_text.json,
# *_wildguard.json and *_overrefusal.json evidence files) and files the pod regenerates, so a
# plain pull aborts -- four times now. Worse, the obvious reflex (`git checkout -- results/`)
# discards ledger rows that exist nowhere else: two were nearly lost on 2026-09-19, one of
# them the run behind the paper's headline number.
#
# TWO DIFFERENT COLLISIONS, and they need different handling:
#
#   1. TRACKED file modified locally. `git checkout --` is right: origin's copy is canonical
#      and the pod's edit is a regeneration. The ledger is the one exception and is merged.
#
#   2. UNTRACKED file that an incoming commit ADDS. This is the pod having produced a result
#      that was then committed from the laptop -- the same file arriving by two routes. Git
#      refuses to overwrite it and aborts the whole pull. This used to be handled by a
#      hardcoded `rm -f results/*_wildguard.json`, i.e. a whitelist that had to be extended
#      by hand for every new script that writes JSON. overrefusal.py was the first one it did
#      not know about, and the pull aborted. So: ASK GIT what is incoming rather than
#      guessing. Each such file is moved aside, not deleted, and after the pull it is
#      compared byte-for-byte with the version that arrived. Identical -> drop the copy.
#      DIFFERENT -> keep it and say so loudly, because that means the pod computed something
#      the committed file does not contain, and only a human can say which is wanted.
#
# THE LEDGER IS RESTORED FROM A TRAP, NOT FROM THE HAPPY PATH. `git checkout -- results/`
# resets the tracked ledger to origin's copy BEFORE the pull is known to succeed, and the
# merge that puts the pod's rows back used to sit AFTER `git pull`. With `set -e`, any pull
# failure exits between those two steps and strands every row the pod added since the last
# push. That is not hypothetical: on 2026-09-19 a pull aborted on an untracked results file
# and SEVEN rows were lost that way -- all six attack.py runs of P1-E7 plus the overrefusal.py
# run, i.e. the entire evidence trail for the experiment, while the .npz outputs survived.
# They were recoverable only because the backup file happened to still be on the pod.
#
# So the restore runs from `trap ... EXIT`: it fires on success, on failure, and on Ctrl-C.
# The backup file is kept either way -- deleting it would re-create the same single point of
# failure one level down.
set -euo pipefail

BACKUP="/workspace/runs.jsonl.podbackup.$(date +%s)"
if [ -f results/runs.jsonl ]; then
    cp results/runs.jsonl "$BACKUP"
    echo "ledger backed up -> $BACKUP  ($(wc -l < "$BACKUP") rows)"
fi

restore_ledger() {
    local rc=$?
    if [ -f "$BACKUP" ]; then
        echo
        python merge_ledger.py "$BACKUP" || echo "  MERGE FAILED -- rows are still in $BACKUP"
        if [ "$rc" -ne 0 ]; then
            echo "  (pull did not complete, but the ledger was restored anyway)"
        fi
    fi
    return $rc
}
trap restore_ledger EXIT

git fetch

UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo origin/main)"

# Untracked files that the incoming commits would add -> move aside so the pull can land.
ASIDE="/workspace/pod_pull_aside.$(date +%s)"
MOVED=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -e "$f" ] || continue
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        mkdir -p "$ASIDE/$(dirname "$f")"
        mv "$f" "$ASIDE/$f"
        echo "moved aside (untracked, incoming): $f"
        MOVED=$((MOVED + 1))
    fi
done < <(git diff --name-only HEAD "$UPSTREAM" -- 2>/dev/null || true)

# Tracked files the pod regenerated: origin's copy wins. The ledger is restored below.
git checkout -- results/ 2>/dev/null || true

git pull


# Now that origin's versions have landed, compare each moved-aside file against them.
if [ "$MOVED" -gt 0 ]; then
    echo
    KEPT=0
    while IFS= read -r saved; do
        rel="${saved#"$ASIDE"/}"
        if [ -f "$rel" ] && cmp -s "$saved" "$rel"; then
            rm -f "$saved"
        else
            echo "  DIFFERS: the pod's $rel is NOT the version that arrived."
            echo "           pod copy kept at $saved -- compare before discarding either."
            KEPT=$((KEPT + 1))
        fi
    done < <(find "$ASIDE" -type f 2>/dev/null)
    find "$ASIDE" -type d -empty -delete 2>/dev/null || true
    if [ "$KEPT" -eq 0 ]; then
        echo "$MOVED moved-aside file(s) matched the committed versions exactly; copies removed."
    else
        echo "$KEPT moved-aside file(s) DIFFER from the committed versions. Nothing was deleted."
    fi
fi

echo
echo "Pull complete. The ledger is merged by the EXIT trap, below."
