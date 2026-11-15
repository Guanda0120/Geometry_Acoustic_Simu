"""
Fault reinsertion: implementation faults of the 2023 thesis code as switches in the verified engine.

Each switch puts one fault back, so that its effect and the checks that reveal it can be measured on its
own (experiment E6, docs/experiment-plan.md). The fault IDs and their definitions follow the audit of the
thesis code (Supplementary Table S5 of the manuscript). Without a switch the engine is unchanged:

    with reinserted('F1'):
        result = simulate(config)

The switches are read where the engine is called from Python (trace_rays, find_image_sources,
rain_fraction, decay_parameters, attenuation_db), not inside the compiled kernels, so a block applies to
everything that is called within it. They are process-wide and not meant for concurrent use.
"""
from contextlib import contextmanager

import numpy as np

FAULTS = {
    'F1': 'ray energy: every ray carries the full source power and a hit is converted with the '
          'spherical-wave law over the whole path, N / (4 pi L2), instead of 1 / (pi r2)',
    'F2': 'scattering-rain solid angle with cos(gamma) = r / d instead of sqrt(1 - (r / d)2)',
    'F3': 'reverberation time from the -5 and -65 dB crossings of the Schroeder curve instead of the '
          'fit from -5 to -35 dB',
    'F6': 'image-source paths not reduced by the specular share (1 - s) of every reflection',
    'F9': 'air attenuation 1e-10 f2 dB/m (classical absorption only) instead of ISO 9613-1',
    'F16': 'image-source path length with the source and the receiver exchanged at the ends of the chain '
           'of reflection points (wrong from the second order upward)',
    'F17': 'tracing stops for all rays at the reflection order at which the strongest ray, with 1 / L2 '
           'spreading, is 60 dB below its level after the first reflection',
}
THESIS_STOP_DB = 60.0  # F17

_active = frozenset()


def active():
    """
    :return: frozenset of the fault IDs that are switched on
    """
    return _active


def on(fault: str):
    """
    :return: Whether a fault is switched on
    """
    return fault in _active


@contextmanager
def reinserted(*faults: str):
    """
    Switch faults on for the duration of a with block (in addition to those already on)
    :param faults: Fault IDs, keys of FAULTS
    """
    global _active
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"Unknown fault(s): {', '.join(sorted(unknown))}; known: {', '.join(FAULTS)}")
    before = _active
    _active = before | frozenset(faults)
    try:
        yield
    finally:
        _active = before


def thesis_attenuation_db(frequency):
    """
    F9: air attenuation of the thesis code, independent of temperature and humidity
    :param frequency: Frequency, Hz, scalar or array
    :return: Attenuation coefficient, dB/m
    """
    return 1e-10 * np.asarray(frequency, dtype=float) ** 2


def thesis_reverberation_time(level: np.ndarray, time_step: float):
    """
    F3: time between the first points of a decay curve at or below -5 and -65 dB
    :param level: (K,) Schroeder curve from the onset, dB; it ends at the last bin with energy, so it always
                  falls below -65 dB there, even if the response is truncated earlier
    :param time_step: Bin width, s
    :return: Reverberation time, s
    """
    return float(np.argmax(level <= -65.0) - np.argmax(level <= -5.0)) * time_step


def thesis_path_length(points: np.ndarray, order: int):
    """
    F16: length of an image-source path with the ends of the chain exchanged
    :param points: (N + 2, 3) source, reflection points in order from the source, receiver
    :param order: Number of reflections
    :return: |receiver - first reflection| + chain + |last reflection - source|, m (the true length for
             orders 0 and 1)
    """
    if order == 0:
        return float(np.linalg.norm(points[1] - points[0]))
    source, chain, receiver = points[0], points[1:order + 1], points[order + 1]
    inner = np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()
    return float(np.linalg.norm(receiver - chain[0]) + inner + np.linalg.norm(chain[-1] - source))


def thesis_stop_order(order_peak: np.ndarray):
    """
    F17: reflection order after which the thesis code stopped tracing
    :param order_peak: (K,) largest ray level E N exp(-m L) / (4 pi L2) over all rays after reflection
                       k + 1 (0 where no ray got that far)
    :return: Number of reflections traced, -1 if the criterion is never met
    """
    if order_peak.size == 0 or order_peak[0] <= 0:
        return -1
    below = np.flatnonzero(order_peak[1:] <= order_peak[0] * 10 ** (-THESIS_STOP_DB / 10))
    return int(below[0]) + 2 if below.size else -1
