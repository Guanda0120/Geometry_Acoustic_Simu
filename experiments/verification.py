"""
E0, known-answer checks (paper Table 2 and Fig. 5; docs/experiment-plan.md).

    uv run python experiments/verification.py [--quick] [--output output/experiments/verification]

Runs the checks V1-V8 of gasim.verification: image sources against the Allen-Berkley lattice, the
lossless box, the fully diffuse box against Kuttruff, the specular box against the exact image lattice,
the energy against the number of rays, the scattering-rain solid angle against Monte Carlo, the decay
estimator on synthetic decays and the air attenuation against ISO 9613-1. Needs no private data. Writes
verification.csv, verification.png and run_info.json to --output; exits 1 if a check fails.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gasim.verification import run_checks

from _common import COLOURS, write_csv, write_run_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='small ray counts and samples (smoke test)')
    parser.add_argument('--output', default='output/experiments/verification')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    checks = run_checks(quick=args.quick)
    write_csv(output / 'verification.csv', [{'id': c.id, 'case': c.case, 'quantity': c.quantity,
                                              'reference': float(c.reference), 'value': float(c.value),
                                              'deviation_pct': 100 * c.deviation, 'criterion': c.criterion,
                                              'passed': c.passed} for c in checks])
    plot(checks, output / 'verification.png')
    write_run_info(output, started, quick=args.quick, checks=len(checks), failed=sum(not c.passed for c in checks))

    print(f"{'id':<4}{'case':<58}{'quantity':<48}{'reference':>12}{'gasim':>12}{'dev %':>9}  pass")
    for c in checks:
        print(f"{c.id:<4}{c.case[:57]:<58}{c.quantity[:47]:<48}{c.reference:>12.5g}{c.value:>12.5g}"
              f"{100 * c.deviation:>9.2f}  {'yes' if c.passed else 'NO'}")
    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed. Wrote {output}")
    return 1 if failed else 0


def plot(checks, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.4, 1]})
    ids = sorted({c.id for c in checks})
    ax = axes[0]
    for x, check_id in enumerate(ids):
        values = [abs(c.deviation) * 100 for c in checks if c.id == check_id and np.isfinite(c.deviation)]
        colours = [COLOURS['new'] if c.passed else '#c0392b' for c in checks
                   if c.id == check_id and np.isfinite(c.deviation)]
        ax.scatter(np.full(len(values), x) + np.linspace(-0.2, 0.2, len(values)), np.maximum(values, 1e-12),
                   c=colours, s=14)
    ax.set_xticks(range(len(ids)), ids)
    ax.set_yscale('log')
    ax.set_ylabel('|deviation from reference| (%)')
    ax.set_title('(a) Every check against its reference', fontsize=10, loc='left')
    ax.grid(alpha=0.3)
    ax = axes[1]
    rain = [c for c in checks if c.id == 'V6']
    for angle, marker in (('0', 'o'), ('45', 's'), ('70', '^')):
        rows = [c for c in rain if f' {angle} deg' in c.case]
        distances = [float(c.case.split(' at ')[1].split(' m')[0]) for c in rows]
        ax.plot(distances, [c.value / c.reference for c in rows], marker + '-', ms=5, label=f'{angle}° from normal')
    ax.axhline(1.0, color=COLOURS['eyring'], lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Distance to the receiver sphere (m)')
    ax.set_ylabel('Scattering rain / Monte Carlo')
    ax.set_title('(b) Scattering-rain solid angle (V6)', fontsize=10, loc='left')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
