#!/usr/bin/env bash
# D1 -- the pre-registered run matrix, in the pre-registered ORDER, resumable.
#
#     ./d1_run.sh controls     # phase 1: 5 control runs -> judge -> calibrate -> stop-check
#     ./d1_run.sh attacks      # phase 2: 5 attack runs  -> judge -> score
#
# WHY TWO PHASES. D1's whole claim is a comparison between two detectors at a matched
# false-positive rate, and the thresholds must come from controls alone. Running every control,
# judging it and calibrating BEFORE any attack run exists makes "frozen before the attacks were
# seen" true in wall-clock time, not only in code. It also puts the stopping rule first: if a
# held-out control loses its steerable layers, coupling tracks fine-tuning per se and the attack
# runs would be wasted compute.
#
# RESUMABLE. A run is skipped if its final dose file already exists, so a pod that dies at run 7
# costs one run, not seven. Partial runs (some doses present, not the last) are re-run from
# dose 0, because the dose curve must come from one continuous training run.
#
# ~22 min per run at 128 tokens on one GPU; ~2 h per phase plus judging.
set -euo pipefail

DOSES=0,50,100,250,500,1000,1500
LAST=1500
mkdir -p logs

run() {  # lineage from tag arm seed
    local lin=$1 src=$2 tag=$3 arm=$4 seed=$5
    local final="results/${tag}_${arm}_s${seed}_dose_${LAST}_refusal_gen128.npz"
    if [ -f "$final" ]; then
        echo "== skip ${tag} ${arm} s${seed} (final dose already on disk)"
        return
    fi
    # A partial run is regenerated from dose 0, so any verdicts it left behind describe text
    # that is about to be overwritten. Delete them rather than trust judge() to notice.
    rm -f results/"${tag}_${arm}_s${seed}"_dose_*_gen128_wildguard.json
    echo "== ${tag} ${arm} s${seed}  ($(date -u +%H:%M:%S) UTC)"
    python dose_response.py --lineage "$lin" --from "$src" --arm "$arm" --seed "$seed" \
        --tag "$tag" --doses "$DOSES" --gen-tokens 128 --skip-ablated-gen \
        2>&1 | tee "logs/${tag}_${arm}_s${seed}.log"
}

judge() {  # arm -- judge every dose file of that arm that has no verdicts yet, WildGuard once
    local arm=$1 todo=()
    for f in results/d1_olmo2_"${arm}"_s*_dose_*_refusal_gen128.npz \
             results/d1_tulu2_"${arm}"_s*_dose_*_refusal_gen128.npz; do
        [ -f "$f" ] || continue
        wg="${f%_refusal_gen128.npz}_gen128_wildguard.json"
        # Skip only a verdict NEWER than its completions; an older one is stale.
        [ -f "$wg" ] && [ "$wg" -nt "$f" ] && continue
        todo+=("$f")
    done
    if [ ${#todo[@]} -eq 0 ]; then echo "== nothing left to judge for ${arm}"; return; fi
    echo "== judging ${#todo[@]} ${arm} file(s), WildGuard loaded once"
    python judge_wildguard.py "${todo[@]}" 2>&1 | tee "logs/judge_${arm}.log"
}

case "${1:-}" in
  controls)
    for s in 1 2 3; do run olmo2     rlvr d1_olmo2 safety-preserved "$s"; done
    for s in 1 2;   do run tulu2_dpo dpo  d1_tulu2 safety-preserved "$s"; done
    judge safety-preserved
    python d1_detect.py calibrate --tags d1_olmo2,d1_tulu2 2>&1 | tee logs/d1_calibrate.log
    # The pre-registered stopping rule, enforced here rather than remembered.
    python - <<'PY'
import json, sys
t = json.load(open("results/d1_thresholds.json"))
fires = t["loo_false_positives"]["steer_frac"]
if fires:
    print(f"\nSTOPPING RULE FIRED: {fires} held-out control(s) lose their steerable layers. "
          "Coupling tracks fine-tuning per se. Do NOT run the attacks; report this.")
    sys.exit(3)
print("\nstopping rule did not fire -- phase 2 may run:  ./d1_run.sh attacks")
PY
    ;;
  attacks)
    if [ ! -f results/d1_thresholds.json ]; then
        echo "results/d1_thresholds.json missing -- run ./d1_run.sh controls first" >&2
        exit 2
    fi
    for s in 1 2 3; do run olmo2     rlvr d1_olmo2 benign "$s"; done
    for s in 1 2;   do run tulu2_dpo dpo  d1_tulu2 benign "$s"; done
    judge benign
    python d1_detect.py score --tags d1_olmo2,d1_tulu2 2>&1 | tee logs/d1_score.log
    ;;
  *)
    echo "usage: $0 controls|attacks" >&2
    exit 1
    ;;
esac
