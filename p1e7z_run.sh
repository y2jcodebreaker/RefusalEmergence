#!/usr/bin/env bash
# P1-E7z -- is the attacked model's re-fit failure a change of DIRECTION or of SIZE?
#
#     ./p1e7z_run.sh              # all three matched pairs (~45-60 min on one GPU)
#     ./p1e7z_run.sh 1            # seed 1 only: the smoke run, ~15 min
#
# Needs the six dose-1500 endpoint adapters that p1e7r_run.sh / d1_run.sh wrote. They are NOT
# regenerable bit-for-bit from here without retraining (dose_response.py, ~28 min each), so
# the script refuses to start rather than silently measuring a different model. PC1 and PC2
# inside p1e7z_strength.py then check that what loaded IS what P1-E7r measured.
set -euo pipefail

SEEDS=${1:-1,2,3}
mkdir -p logs
missing=0
for s in ${SEEDS//,/ }; do
    for d in "models/p1e7r_olmo2-benign_s${s}-adapter-1500" \
             "models/d1_olmo2-safety-preserved_s${s}-adapter-1500"; do
        [ -f "$d/adapter_model.safetensors" ] || { echo "missing adapter: $d" >&2; missing=1; }
    done
done
if [ "$missing" -ne 0 ]; then
    echo "restore the adapters (tar xzf p1e7-adapters-all.tgz) or retrain with p1e7r_run.sh" >&2
    exit 2
fi

python p1e7z_strength.py --seeds "$SEEDS" 2>&1 | tee "logs/p1e7z_s${SEEDS//,/}.log"
