#!/usr/bin/env bash
# Run experiments one after another (docs/experiment-plan.md), from any directory:
#
#   experiments/run_batch.sh batch1                       # E0 verification, E1 validation, E2 ray count
#   experiments/run_batch.sh batch2                       # E3 input sensitivity, E4 scenarios
#   experiments/run_batch.sh all                          # batch1 and batch2
#   experiments/run_batch.sh verification scenarios       # single scripts by name
#
# Every script runs with its defaults and writes to output/experiments/<its folder>/; its console output goes
# to output/experiments/<script>.log, and output/experiments/progress.log records the start, end and exit
# code of every step. On macOS the batch keeps the computer awake (caffeinate); everywhere it runs at low
# priority (nice), so the machine stays usable. A failing step does not stop the batch; the exit code is
# that of the last failure.
set -u
cd "$(dirname "$0")/.." || exit 1

expand() {
    case "$1" in
        batch1) echo "verification validation_rooms ray_count" ;;
        batch2) echo "input_sensitivity scenarios" ;;
        all) echo "verification validation_rooms ray_count input_sensitivity scenarios" ;;
        *) echo "$1" ;;
    esac
}

if [ $# -eq 0 ]; then
    sed -n '2,13p' "$0"
    exit 2
fi

steps=""
for argument in "$@"; do
    steps="$steps $(expand "$argument")"
done
for step in $steps; do
    if [ ! -f "experiments/$step.py" ]; then
        echo "unknown experiment: $step (no experiments/$step.py)" >&2
        exit 2
    fi
done

awake=""
if command -v caffeinate > /dev/null 2>&1; then
    awake="caffeinate -i"
fi

mkdir -p output/experiments
progress=output/experiments/progress.log
status=0
echo "$(date '+%F %T') batch start: $steps (commit $(git rev-parse --short HEAD 2> /dev/null || echo unknown))" >> "$progress"
for step in $steps; do
    echo "$(date '+%F %T') start $step" >> "$progress"
    $awake nice -n 10 uv run python "experiments/$step.py" > "output/experiments/$step.log" 2>&1
    code=$?
    echo "$(date '+%F %T') end   $step, exit $code" >> "$progress"
    if [ $code -ne 0 ]; then
        status=$code
    fi
done
echo "$(date '+%F %T') batch end, exit $status" >> "$progress"
exit $status
