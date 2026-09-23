#!/usr/bin/env bash
# P1-E7r -- three OLMo 2 attack seeds matched to D1's controls, then the verdict.
#
#     ./p1e7r_run.sh
#
# Attack seed k shares its Alpaca subset and LoRA init with D1 control seed k (same --seed), so
# the controls D1 already ran are the matched arm. Same protocol as D1 exactly: doses, 128-token
# generations, held-out evaluation half, frozen dose-0 direction. Resumable: a run whose final
# dose is on disk is skipped, and a regenerated run deletes its own stale verdicts first.
# ~28 min per run, ~1.5 h plus judging.
set -euo pipefail

DOSES=0,50,100,250,500,1000,1500
LAST=1500
TAG=p1e7r_olmo2
mkdir -p logs

for f in results/d1_olmo2_safety-preserved_s{1,2,3}_dose_${LAST}_refusal_gen128.npz; do
    [ -f "$f" ] || { echo "matched control missing: $f -- pull the D1 controls first" >&2; exit 2; }
done

for s in 1 2 3; do
    final="results/${TAG}_benign_s${s}_dose_${LAST}_refusal_gen128.npz"
    if [ -f "$final" ]; then echo "== skip seed ${s} (final dose on disk)"; continue; fi
    rm -f results/"${TAG}_benign_s${s}"_dose_*_gen128_wildguard.json
    echo "== ${TAG} benign s${s}  ($(date -u +%H:%M:%S) UTC)"
    python dose_response.py --lineage olmo2 --from rlvr --arm benign --seed "$s" \
        --tag "$TAG" --doses "$DOSES" --gen-tokens 128 --skip-ablated-gen --experiment P1-E7r \
        2>&1 | tee "logs/${TAG}_benign_s${s}.log"
done

todo=()
for f in results/${TAG}_benign_s*_dose_*_refusal_gen128.npz; do
    [ -f "$f" ] || continue
    wg="${f%_refusal_gen128.npz}_gen128_wildguard.json"
    [ -f "$wg" ] && [ "$wg" -nt "$f" ] && continue
    todo+=("$f")
done
if [ ${#todo[@]} -gt 0 ]; then
    echo "== judging ${#todo[@]} file(s), WildGuard loaded once"
    python judge_wildguard.py "${todo[@]}" 2>&1 | tee logs/judge_p1e7r.log
fi

python p1e7r_check.py 2>&1 | tee logs/p1e7r_check.log
