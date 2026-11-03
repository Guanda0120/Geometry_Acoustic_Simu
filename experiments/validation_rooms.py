"""
E1, field validation of the three rooms (paper Table 3, Fig. 6 and the receiver tables; docs/experiment-plan.md).

    uv run python experiments/validation_rooms.py [--seeds 10] [--rays-per-m3 100] [--scenario-seeds 5]

Base cases use only the documented inputs (configs/<room>_paper.toml: Appendix A materials, alpha_d used
directly, ISO 9613-1 air at 20 degC / 50 % RH). Every case is run with --seeds seeds; the room value of a
run is the median over receivers, reported as mean, standard deviation and range over the seeds. The
undocumented scattering coefficient is shown as a band: the room is rerun with a uniform scattering
coefficient at both ends of SCATTERING_RANGE (--scenario-seeds seeds each). Sabine and Eyring from the
same model and inputs are the baselines. Measured room values are the mean over positions with a 95 %
confidence interval; errors are given in % and in JND (5 %).

Needs the private models (models/private/) and the measured data outside the repository (../data, see
its README). Writes summary.csv, receivers.csv, validation.png and run_info.json to --output.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.bands import OCTAVE_BANDS
from gasim.config import load_config

from _common import (
    BAND_LABELS,
    COLOURS,
    JND,
    PAPER_ROOMS,
    SCATTERING_RANGE,
    cell,
    eyring_of,
    measured_statistics,
    read_receiver_table,
    room_value,
    run_seeds,
    sabine_of,
    uniform_scattering,
    write_csv,
    write_run_info,
)

CASES = PAPER_ROOMS


def standardised_rmse(simulated, measured):
    """
    JND-standardised RMS error over receivers, sqrt(mean(e_i^2)) / JND, for every seed and band
    :param simulated: (S, R, B); measured: (R, B)
    :return: (S, B)
    """
    errors = (simulated - measured[None]) / measured[None]
    return np.sqrt(np.nanmean(errors ** 2, axis=1)) / JND


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../data', help='folder with measured/ and thesis_simulated/')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--rays-per-m3', type=float, default=100.0)
    parser.add_argument('--scenario-seeds', type=int, default=5, help='seeds per end of the scattering band')
    parser.add_argument('--no-scattering-band', action='store_true', help='skip the scattering scenarios')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--output', default='output/experiments/validation')
    args = parser.parse_args(argv)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    summary, receivers, results = [], [], {}
    for case in args.cases:
        title, config_path, stem = CASES[case]
        config = load_config(config_path)
        if not config.model.exists():
            print(f"skip {case}: private model not available")
            continue
        measured = read_receiver_table(data / 'measured' / f'{stem}.csv')
        measured_mean, measured_ci = measured_statistics(measured)
        thesis = read_receiver_table(data / 'thesis_simulated' / f'{stem}.csv')
        base = run_seeds(config, args.seeds, args.rays_per_m3, case)
        band = None
        if not args.no_scattering_band:
            band = np.array([room_value(run_seeds(uniform_scattering(config, s), args.scenario_seeds,
                                                 args.rays_per_m3, f'{case} s = {s}')['t30']).mean(axis=0)
                             for s in SCATTERING_RANGE[case]])
        sabine, eyring = sabine_of(config), eyring_of(config)
        t30 = room_value(base['t30'])  # (S, B)
        results[case] = (title, measured_mean, measured_ci, thesis, t30, band, sabine, eyring)

        srmse = (standardised_rmse(base['t30'], measured) if measured is not None
                 and measured.shape[0] == base['t30'].shape[1] else None)
        reference_srmse = standardised_rmse(np.broadcast_to(measured_mean, measured.shape)[None], measured)[0] \
            if srmse is not None else None
        for b, frequency in enumerate(OCTAVE_BANDS):
            row = {'case': case, 'band': int(frequency), 'rays': base['rays'], 'seeds': args.seeds,
                   't30_mean': t30[:, b].mean(),
                   't30_sd': t30[:, b].std(ddof=1) if args.seeds > 1 else float('nan'),
                   't30_min': t30[:, b].min(), 't30_max': t30[:, b].max(),
                   't20_mean': room_value(base['t20'])[:, b].mean(), 'edt_mean': room_value(base['edt'])[:, b].mean(),
                   'c50_mean': room_value(base['c50'])[:, b].mean(), 'sabine': sabine[b], 'eyring': eyring[b]}
            if band is not None:
                row.update({f't30_s{s:g}': band[i, b] for i, s in enumerate(SCATTERING_RANGE[case])})
            if measured_mean is not None:
                row.update({'measured': measured_mean[b], 'measured_ci95': measured_ci[b]})
                for name in ('t30_mean', 'sabine', 'eyring'):
                    row[f'error_{name.split("_")[0]}_pct'] = 100 * (row[name] / measured_mean[b] - 1)
                row['error_t30_jnd'] = row['error_t30_pct'] / (100 * JND)
            if srmse is not None:
                row.update({'srmse_jnd': srmse[:, b].mean(), 'srmse_reference_jnd': reference_srmse[b]})
            if thesis is not None:
                row['thesis'] = thesis[:, b].mean()
            summary.append(row)
            for r in range(base['t30'].shape[1]):
                receivers.append({'case': case, 'receiver': r + 1, 'band': int(frequency),
                                  't30_mean': np.nanmean(base['t30'][:, r, b]),
                                  't30_sd': np.nanstd(base['t30'][:, r, b], ddof=1) if args.seeds > 1 else '',
                                  'measured': measured[r, b] if measured is not None and r < len(measured) else ''})

    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'receivers.csv', receivers)
    plot(results, output / 'validation.png')
    write_run_info(output, started, cases={c: CASES[c][1] for c in args.cases}, seeds=args.seeds,
                   rays_per_m3=args.rays_per_m3, scenario_seeds=args.scenario_seeds,
                   scattering_range={c: SCATTERING_RANGE[c] for c in args.cases} if not args.no_scattering_band
                   else None)
    print_summary(summary)
    print(f"\nWrote {output}")
    return 0


def print_summary(summary):
    print(f"\n{'case':<13}{'band':>6}{'meas':>7}{'±CI':>6}{'T30':>7}{'±sd':>7}{'T20':>7}{'EDT':>7}{'Sabine':>8}"
          f"{'Eyring':>8}{'err %':>8}{'JND':>6}{'sRMSE':>7}")
    for row in summary:
        print(f"{row['case']:<13}{row['band']:>6}{cell(row, 'measured', 7)}{cell(row, 'measured_ci95', 6)}"
              f"{cell(row, 't30_mean', 7)}{cell(row, 't30_sd', 7, '.3f')}{cell(row, 't20_mean', 7)}"
              f"{cell(row, 'edt_mean', 7)}{cell(row, 'sabine', 8)}{cell(row, 'eyring', 8)}"
              f"{cell(row, 'error_t30_pct', 8, '+.1f')}{cell(row, 'error_t30_jnd', 6, '+.1f')}"
              f"{cell(row, 'srmse_jnd', 7, '.1f')}")


def plot(results, path: Path):
    n = max(len(results), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    x = np.arange(len(OCTAVE_BANDS))
    for ax, (title, measured, ci, thesis, t30, band, sabine, eyring) in zip(axes[0], results.values()):
        if band is not None:
            ax.fill_between(x, band.min(axis=0), band.max(axis=0), color=COLOURS['new'], alpha=0.15, lw=0,
                            label='Scattering range')
        if measured is not None:
            ax.errorbar(x, measured, yerr=ci, fmt='o-', color=COLOURS['measured'], capsize=3, label='Measured (95% CI)')
        if thesis is not None:
            ax.plot(x, thesis.mean(axis=0), 's--', color=COLOURS['thesis'], ms=4, label='Thesis code')
        ax.errorbar(x, t30.mean(axis=0), yerr=t30.std(axis=0, ddof=1) if len(t30) > 1 else None, fmt='^-',
                    color=COLOURS['new'], capsize=3, label='gasim T30 (mean ± sd over seeds)')
        ax.plot(x, sabine, ':', color=COLOURS['eyring'], label='Sabine')
        ax.plot(x, eyring, '--', color=COLOURS['eyring'], lw=0.8, label='Eyring')
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel('Octave band (Hz)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('Reverberation time (s)')
    handles = {label: handle for ax in axes[0] for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
