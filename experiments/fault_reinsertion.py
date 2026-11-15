"""
E6, fault reinsertion: which checks reveal which faults (paper Table 5 and Fig. 10; docs/experiment-plan.md).

    uv run python experiments/fault_reinsertion.py [--seeds 20] [--rays-per-m3 100] [--cases classroom ...]
                                                   [--faults F1 ...] [--quick-checks]

Each fault of the thesis code (gasim.faults) is switched back on alone in the verified engine. The mutant
then meets two kinds of evidence:

1. the known-answer checks V1-V8 of gasim.verification, with their own pass criteria;
2. the measured rooms with their documented inputs (configs/<room>_paper.toml), run with the same seeds
   as the baseline, so that mutant and baseline are paired.

The criteria were fixed before any mutant was run (audit of the thesis code, CRITERIA.md of 2026-10-25);
M is the mutant, B the baseline, both the mean over seeds of the median over receivers, sigma the standard
deviation of the baseline over seeds, JND 5 % for T30 and EDT and 1 dB for C50:

    effect            |M - B| > max(1 JND, 2 sigma) in at least 2 of the 6 bands
    measured RT       pass: |T30 - measured| <= 10 % in at least 3 of the 4 bands 500 Hz - 4 kHz. A fault is
                      detected if the baseline passes and the mutant fails; if the baseline fails, the room
                      is not informative
    perfect input     |M - B| > max(2 JND, 2 sigma) in at least 2 bands, for T30, EDT and C50: what a
                      comparison could reveal at best, if the inputs were exact
    known-answer      the mutant fails a check that the baseline passes

A value that cannot be evaluated (nan) counts as a difference and as a failed comparison.

Writes checks.csv, effects.csv, matrix.csv, fault_reinsertion.png and run_info.json to --output.
"""
import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config
from gasim.faults import FAULTS, reinserted
from gasim.geometry import load_room
from gasim.verification import run_checks

from _common import (
    BAND_LABELS,
    PAPER_ROOMS,
    decay_over_seeds,
    measured_statistics,
    read_receiver_table,
    room_value,
    with_decay_time,
    with_settings,
    write_csv,
    write_run_info,
)

METRICS = {'t30': 'T30', 'edt': 'EDT', 'c50': 'C50'}
JND = {'t30': 0.05, 'edt': 0.05, 'c50': 1.0}  # relative for T30 and EDT, dB for C50 (ISO 3382-1)
RELATIVE = {'t30': True, 'edt': True, 'c50': False}
MIN_BANDS = 2  # bands in which a difference must exceed its threshold
MEASURED_BANDS = [2, 3, 4, 5]  # 500 Hz - 4 kHz
MEASURED_TOLERANCE = 0.10  # 2 JND
MEASURED_MIN_BANDS = 3
BASELINE = 'none'


def thresholds(baseline: np.ndarray, metric: str, jnds: float):
    """
    :param baseline: (S, B) room values of the baseline per seed
    :param jnds: Number of JNDs (1 for an effect, 2 for detection with perfect input)
    :return: (B,) max(jnds x JND, 2 sigma), in the unit of the metric; sigma is 0 for a single seed
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(baseline, axis=0)
        sigma = np.nanstd(baseline, axis=0, ddof=1) if len(baseline) > 1 else np.zeros_like(mean)
    jnd = JND[metric] * np.abs(mean) if RELATIVE[metric] else np.full_like(mean, JND[metric])
    return np.fmax(jnds * jnd, 2 * np.nan_to_num(sigma))


def effect(baseline: np.ndarray, mutant: np.ndarray, metric: str):
    """
    Paired difference of a mutant against the baseline
    :param baseline, mutant: (S, B) room values per seed, the same seeds in the same order
    :return: dict with difference (B,), the thresholds for 1 and 2 JND and whether each is exceeded (B,)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        difference = np.nanmean(mutant, axis=0) - np.nanmean(baseline, axis=0)
    result = {'difference': difference}
    for jnds in (1, 2):
        limit = thresholds(baseline, metric, jnds)
        result[f'threshold_{jnds}'] = limit
        result[f'exceeds_{jnds}'] = ~(np.abs(difference) <= limit)  # nan counts as exceeding
    return result


def passes_measured(t30: np.ndarray, measured: np.ndarray):
    """
    :param t30: (B,) simulated room value
    :param measured: (B,) measured room mean
    :return: Whether the comparison with the measurement passes (criterion "measured RT")
    """
    within = np.abs(t30[MEASURED_BANDS] / measured[MEASURED_BANDS] - 1) <= MEASURED_TOLERANCE
    return int(within.sum()) >= MEASURED_MIN_BANDS


def measured_verdict(baseline_passes: bool, mutant_passes: bool):
    if not baseline_passes:
        return 'not informative'
    return 'not detected' if mutant_passes else 'detected'


def check_summary(checks):
    """
    :param checks: List of gasim.verification.Check
    :return: {check id: (number of comparisons, number failed, largest |deviation| in %)}
    """
    summary = {}
    for check in checks:
        count, failed, worst = summary.get(check.id, (0, 0, 0.0))
        deviation = abs(check.deviation) * 100
        summary[check.id] = (count + 1, failed + (not check.passed),
                             max(worst, deviation) if math.isfinite(deviation) else worst)
    return summary


def known_answer_verdict(baseline, mutant):
    """
    :param baseline, mutant: (comparisons, failed, worst) of one check
    :return: 'detected', 'not detected' or 'not informative' (the baseline itself fails)
    """
    if baseline[1]:
        return 'not informative'
    return 'detected' if mutant[1] else 'not detected'


def run_room(case: str, faults, seeds: int, rays_per_m3: float, data: Path):
    """
    Baseline and mutants of one room with the same seeds, rays and trace length
    :return: dict {fault or BASELINE: {metric: (S, B) room values}}, measured mean (B,) or None, rays
    """
    _, config_path, stem = PAPER_ROOMS[case]
    config = load_config(config_path)
    room, _ = load_room(config)
    config = with_settings(with_decay_time(config), rays=math.ceil(rays_per_m3 * room.volume))
    measured, _ = measured_statistics(read_receiver_table(data / 'measured' / f'{stem}.csv'))
    runs = {}
    for fault in [BASELINE] + list(faults):
        with reinserted(*([] if fault == BASELINE else [fault])):
            run = decay_over_seeds(config, seeds, f'{case}: {fault}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            runs[fault] = {metric: room_value(run[metric]) for metric in METRICS}
    return runs, measured, config.simulation.rays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--cases', nargs='*', choices=list(PAPER_ROOMS), default=['classroom', 'lecture_hall'])
    parser.add_argument('--faults', nargs='+', choices=list(FAULTS), default=list(FAULTS))
    parser.add_argument('--quick-checks', action='store_true', help='small known-answer checks (smoke test)')
    parser.add_argument('--skip-checks', action='store_true', help='rooms only')
    parser.add_argument('--output', default='output/experiments/fault_reinsertion')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    matrix = {fault: {'fault': fault, 'description': FAULTS[fault]} for fault in args.faults}

    # 1. Known-answer checks
    check_rows, check_ids = [], []
    if not args.skip_checks:
        summaries = {}
        for fault in [BASELINE] + args.faults:
            with reinserted(*([] if fault == BASELINE else [fault])):
                summaries[fault] = check_summary(run_checks(quick=args.quick_checks))
            print(f"  known-answer checks: {fault} done", flush=True)
        check_ids = sorted(summaries[BASELINE])
        for fault, summary in summaries.items():
            for check_id in check_ids:
                count, failed, worst = summary[check_id]
                row = {'fault': fault, 'check': check_id, 'comparisons': count, 'failed': failed,
                       'largest_deviation_pct': float(worst)}
                if fault != BASELINE:
                    row['verdict'] = known_answer_verdict(summaries[BASELINE][check_id], summary[check_id])
                    matrix[fault][check_id] = row['verdict']
                check_rows.append(row)
        write_csv(output / 'checks.csv', check_rows)

    # 2. Measured rooms
    effect_rows, curves, rays = [], {}, {}
    for case in args.cases:
        if not load_config(PAPER_ROOMS[case][1]).model.exists():
            print(f"skip {case}: private model not available")
            continue
        runs, measured, rays[case] = run_room(case, args.faults, args.seeds, args.rays_per_m3, data)
        baseline = runs[BASELINE]
        baseline_mean = {metric: np.nanmean(baseline[metric], axis=0) for metric in METRICS}
        baseline_passes = measured is not None and passes_measured(baseline_mean['t30'], measured)
        curves[case] = {}
        for fault in args.faults:
            mutant = runs[fault]
            for metric, name in METRICS.items():
                result = effect(baseline[metric], mutant[metric], metric)
                matrix[fault][f'{case}: effect {name}'] = 'yes' if result['exceeds_1'].sum() >= MIN_BANDS else 'no'
                matrix[fault][f'{case}: perfect input {name}'] = \
                    'yes' if result['exceeds_2'].sum() >= MIN_BANDS else 'no'
                for b, band in enumerate(OCTAVE_BANDS):
                    difference = float(result['difference'][b])
                    effect_rows.append({
                        'case': case, 'fault': fault, 'metric': name, 'band': int(band),
                        'baseline': float(baseline_mean[metric][b]), 'mutant': float(baseline_mean[metric][b]) + difference,
                        'difference': difference,
                        'difference_pct': 100 * difference / float(baseline_mean[metric][b]) if RELATIVE[metric]
                        else float('nan'),
                        'threshold_1jnd': float(result['threshold_1'][b]),
                        'threshold_2jnd': float(result['threshold_2'][b]),
                        'exceeds_1jnd': bool(result['exceeds_1'][b]), 'exceeds_2jnd': bool(result['exceeds_2'][b])})
            mutant_t30 = np.nanmean(mutant['t30'], axis=0)
            curves[case][fault] = 100 * (mutant_t30 / baseline_mean['t30'] - 1)
            if measured is not None:
                matrix[fault][f'{case}: measured RT'] = measured_verdict(baseline_passes,
                                                                        passes_measured(mutant_t30, measured))
                matrix[fault][f'{case}: seeds passing, baseline'] = \
                    float(np.mean([passes_measured(row, measured) for row in baseline['t30']]))
                matrix[fault][f'{case}: seeds passing, mutant'] = \
                    float(np.mean([passes_measured(row, measured) for row in mutant['t30']]))

    write_csv(output / 'effects.csv', effect_rows)
    write_csv(output / 'matrix.csv', list(matrix.values()))
    plot(matrix, check_ids, curves, output / 'fault_reinsertion.png')
    write_run_info(output, started, cases={c: PAPER_ROOMS[c][1] for c in curves}, rays=rays, seeds=args.seeds,
                   rays_per_m3=args.rays_per_m3, faults={f: FAULTS[f] for f in args.faults},
                   quick_checks=args.quick_checks,
                   criteria={'jnd': JND, 'min_bands': MIN_BANDS, 'measured_bands_hz': OCTAVE_BANDS[MEASURED_BANDS].tolist(),
                             'measured_tolerance': MEASURED_TOLERANCE, 'measured_min_bands': MEASURED_MIN_BANDS})

    columns = [key for key in next(iter(matrix.values())) if key not in ('fault', 'description')
               and 'seeds passing' not in key and ': effect' not in key]
    print('\n' + f"{'fault':<6}" + ''.join(f'{shorten(column):>12}' for column in columns))
    for fault, row in matrix.items():
        print(f'{fault:<6}' + ''.join(f"{SHORT.get(row.get(column), '-'):>12}" for column in columns))
    print(f"\nWrote {output}")
    return 0


SHORT = {'detected': 'DETECTED', 'not detected': 'no', 'not informative': 'n/i', 'yes': 'YES', 'no': 'no'}


def shorten(column: str):
    return (column.replace('lecture_hall', 'LH').replace('classroom', 'CR').replace('mf_hall', 'MF')
            .replace('perfect input', 'PI').replace('measured RT', 'meas').replace(': ', ' '))


def plot(matrix, check_ids, curves, path: Path):
    """
    (a) faults x checks, (b...) change of T30 by each fault per room
    """
    cases = list(curves)
    columns = [(check_id, check_id) for check_id in check_ids]
    for case in cases:
        title = PAPER_ROOMS[case][0]
        if any(f'{case}: measured RT' in row for row in matrix.values()):
            columns.append((f'{case}: measured RT', f'{title}\nmeasured RT'))
        columns += [(f'{case}: perfect input {name}', f'{title}\nexact input, {name}') for name in METRICS.values()]
    colour = {'detected': '#c0392b', 'yes': '#c0392b', 'not detected': '#ecf0f1', 'no': '#ecf0f1',
              'not informative': '#f7dc6f'}
    faults = list(matrix)
    fig, axes = plt.subplots(1, 1 + len(cases), figsize=(max(7.0, 0.62 * len(columns) + 1.5) + 4.2 * len(cases), 4.2),
                             gridspec_kw={'width_ratios': [max(1.6, 0.16 * len(columns))] + [1] * len(cases)},
                             squeeze=False)
    ax = axes[0][0]
    for y, fault in enumerate(faults):
        for x, (key, _) in enumerate(columns):
            verdict = matrix[fault].get(key)
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=colour.get(verdict, 'white'), edgecolor='white', lw=1.5))
    ax.set_xlim(0, max(len(columns), 1))
    ax.set_ylim(len(faults), 0)
    ax.set_xticks(np.arange(len(columns)) + 0.5, [label for _, label in columns], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(faults)) + 0.5, faults)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in ('#c0392b', '#ecf0f1', '#f7dc6f')]
    ax.legend(handles, ['fault revealed', 'not revealed', 'baseline fails: not informative'], fontsize=7,
              loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    ax.set_title('(a) Which evidence reveals which fault', fontsize=10, loc='left')

    x = np.arange(len(OCTAVE_BANDS))
    for ax, case, letter in zip(axes[0][1:], cases, 'bcd'):
        ax.axhspan(-100 * MEASURED_TOLERANCE, 100 * MEASURED_TOLERANCE, color='grey', alpha=0.15, label='±2 JND')
        for fault, change in curves[case].items():
            ax.plot(x, change, 'o-', ms=3, lw=1, label=fault)
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_ylabel('Change of T30 by the fault (%)')
        ax.set_title(f'({letter}) {PAPER_ROOMS[case][0]}', fontsize=10, loc='left')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
