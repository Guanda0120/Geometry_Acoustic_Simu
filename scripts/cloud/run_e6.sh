#!/usr/bin/env bash
# E6, fault reinsertion. Run from the repository root on Linux after uv sync --frozen --no-dev.
# DEADLINE_EPOCH must precede the cloud instance release time by at least 10 min.
# This stops computation; only the cloud release schedule stops instance billing.
# CASES selects the rooms (default: classroom lecture_hall, as in docs/experiment-plan.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DEADLINE_EPOCH:?Set an absolute computation deadline in Unix seconds}"
CASES="${CASES:-classroom lecture_hall}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-32}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
PYTHON=.venv/bin/python
CASES="$CASES" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from gasim.config import load_config
for case in os.environ['CASES'].split():
    config = load_config(f'configs/{case}_paper.toml')
    for path in (config.model, Path(f'../data/measured/{case}.csv')):
        if not path.is_file():
            raise SystemExit(f'Missing required input: {path}')
import numba, rhino3dm
from gasim.faults import FAULTS
print('Input and import checks passed; Numba threads:', numba.get_num_threads(), '; faults:', ' '.join(FAULTS))
PY
run_dir="output/cloud/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
printf '%s\n' "$run_dir" > output/cloud/latest-run.txt
printf 'experiment\tcase\texit_code\tfinished_utc\n' > "$run_dir/status.tsv"

# One step for the known-answer checks, then one per room, so every finished step is kept if a later one
# reaches the deadline. Scientific defaults: 20 seeds, 100 rays/m3, full-size checks.
step() {
    local name="$1"
    shift
    local remaining=$((DEADLINE_EPOCH - $(date +%s)))
    if (( remaining <= 0 )); then exit 124; fi
    local destination="$run_dir/fault_reinsertion/$name"
    mkdir -p "$destination"
    local result=0
    timeout --signal=TERM --kill-after=15s "${remaining}s" \
        "$PYTHON" experiments/fault_reinsertion.py "$@" --seeds 20 --rays-per-m3 100 \
        --data ../data --output "$destination" > "$destination/run.log" 2>&1 || result=$?
    printf 'fault_reinsertion\t%s\t%s\t%s\n' "$name" "$result" "$(date -u +%FT%TZ)" >> "$run_dir/status.tsv"
    if (( result != 0 )); then exit "$result"; fi
}

step checks --cases
for room in $CASES; do
    step "$room" --cases "$room" --skip-checks
done
printf 'E6 completed: checks and %s.\n' "$CASES" | tee "$run_dir/COMPLETE.txt"
