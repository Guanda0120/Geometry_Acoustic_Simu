#!/usr/bin/env bash
# Run from the repository root on Linux after uv sync --frozen --no-dev.
# DEADLINE_EPOCH must precede the cloud instance release time by at least 10 min.
# This stops computation; only the cloud release schedule stops instance billing.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DEADLINE_EPOCH:?Set an absolute computation deadline in Unix seconds}"
export NUMBA_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
PYTHON=.venv/bin/python
"$PYTHON" - <<'PY'
from pathlib import Path
from gasim.config import load_config
for case in ('classroom', 'lecture_hall', 'mf_hall'):
    config = load_config(f'configs/{case}_paper.toml')
    for path in (config.model, Path(f'../data/measured/{case}.csv')):
        if not path.is_file():
            raise SystemExit(f'Missing required input: {path}')
import numba, rhino3dm
assert numba.get_num_threads() == 32
print('Input and import checks passed; Numba threads:', numba.get_num_threads())
PY
run_dir="output/cloud/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
printf '%s\n' "$run_dir" > output/cloud/latest-run.txt
printf 'experiment\tcase\texit_code\tfinished_utc\n' > "$run_dir/status.tsv"
# E4 first; retain each completed room even if a later room reaches the deadline.
# Scientific defaults remain 5 seeds and 100 rays/m3, with no reduced workload.
for experiment in scenarios input_sensitivity; do
    for room in classroom lecture_hall mf_hall; do
        remaining=$((DEADLINE_EPOCH - $(date +%s)))
        if (( remaining <= 0 )); then exit 124; fi
        destination="$run_dir/$experiment/$room"
        mkdir -p "$destination"
        result=0
        timeout --signal=TERM --kill-after=15s "${remaining}s" \
            "$PYTHON" "experiments/$experiment.py" \
            --cases "$room" --seeds 5 --rays-per-m3 100 \
            --data ../data --output "$destination" \
            > "$destination/run.log" 2>&1 || result=$?
        printf '%s\t%s\t%s\t%s\n' "$experiment" "$room" "$result" \
            "$(date -u +%FT%TZ)" >> "$run_dir/status.tsv"
        if (( result != 0 )); then exit "$result"; fi
    done
done
printf 'All E3 and E4 rooms completed.\n' | tee "$run_dir/COMPLETE.txt"
