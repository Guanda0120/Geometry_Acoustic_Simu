"""
Decay curves and reverberation parameters after ISO 3382-1/-2.

Input is an energy histogram: energy per time bin, time along the last axis. The decay curve is the
Schroeder backward integral, evaluated from the first bin that holds energy (the direct sound).
"""
from dataclasses import dataclass

import numpy as np

# Evaluation ranges in dB below the start of the decay curve
EVALUATION_RANGES = {'EDT': (0.0, -10.0), 'T20': (-5.0, -25.0), 'T30': (-5.0, -35.0)}
# The curve must continue this far below the evaluation range, so truncation of the response does not
# bend the fitted part: a single-slope decay truncated 10 dB below the range gives T30 0.85 % short,
# 15 dB below gives 0.3 % (ISO 3382-2 asks 10 dB above the noise floor for measurements)
DYNAMIC_RANGE_MARGIN = 15.0  # dB


def schroeder_curve(energy: np.ndarray):
    """
    Backward-integrated decay level (Schroeder 1965)
    :param energy: (..., K) energy per time bin
    :return: (..., K) level in dB re the total energy; -inf after the last bin with energy, nan if there is none
    """
    tail = np.cumsum(energy[..., ::-1], axis=-1)[..., ::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(tail / tail[..., :1])


@dataclass(frozen=True)
class DecayFit:
    """
    Least-squares line through one decay curve within an evaluation range
    """
    value: float  # s, time for a 60 dB decay; nan if the range cannot be evaluated
    slope: float  # dB/s
    r: float  # correlation coefficient of the fit
    start: float  # s, first point of the fit, re the onset
    stop: float  # s, last point of the fit, re the onset
    reason: str = ''  # why value is nan

    @property
    def nonlinearity(self):
        """
        Non-linearity parameter xi = 1000 (1 - r^2), in per mille (ISO 3382-2 Annex B)
        """
        return 1000 * (1 - self.r ** 2)


def fit_decay(energy: np.ndarray, time_step: float, upper: float, lower: float):
    """
    Fit one decay curve between two levels
    :param energy: (K,) energy per time bin
    :param time_step: Bin width, s
    :param upper: Upper level of the evaluation range, dB (e.g. -5)
    :param lower: Lower level of the evaluation range, dB (e.g. -35)
    :return: DecayFit
    """
    nan = float('nan')
    filled = np.flatnonzero(energy > 0)
    if filled.size == 0:
        return DecayFit(nan, nan, nan, nan, nan, 'no energy')
    level = schroeder_curve(energy[filled[0]:filled[-1] + 1])
    time = np.arange(level.size) * time_step
    in_range = (level <= upper) & (level >= lower)
    if level[-1] > lower or in_range.sum() < 3:
        return DecayFit(nan, nan, nan, nan, nan, f'decay curve does not reach {lower:.0f} dB')

    t, y = time[in_range], level[in_range]
    slope, _ = np.polyfit(t, y, 1)
    r = np.corrcoef(t, y)[0, 1]
    # Depth the fitted decay reaches by the end of the response; the truncation bias depends on how far
    # this lies below the evaluation range (the Schroeder level of the last bin does not tell)
    depth = -slope * level.size * time_step
    if depth < -lower + DYNAMIC_RANGE_MARGIN:
        return DecayFit(nan, slope, r, t[0], t[-1],
                        f'decay is {depth:.0f} dB deep, needs {-lower + DYNAMIC_RANGE_MARGIN:.0f} dB')
    return DecayFit(-60.0 / slope, slope, r, t[0], t[-1])


@dataclass(frozen=True)
class DecayParameters:
    """
    Reverberation parameters for every curve of an energy histogram, arrays of shape energy.shape[:-1]
    """
    edt: np.ndarray  # s
    t20: np.ndarray  # s
    t30: np.ndarray  # s
    xi_t20: np.ndarray  # per mille
    xi_t30: np.ndarray  # per mille

    @property
    def curvature(self):
        """
        C = 100 (T30 / T20 - 1) in %; large values point to a curved (e.g. double-slope) decay (ISO 3382-2 Annex B)
        """
        return 100 * (self.t30 / self.t20 - 1)


def decay_parameters(energy: np.ndarray, time_step: float):
    """
    EDT, T20 and T30 of every decay curve in a histogram
    :param energy: (..., K) energy per time bin, e.g. (receivers, bands, bins)
    :param time_step: Bin width, s
    :return: DecayParameters
    """
    energy = np.asarray(energy, dtype=float)
    curves = energy.reshape(-1, energy.shape[-1])
    fits = {name: [fit_decay(curve, time_step, *limits) for curve in curves]
            for name, limits in EVALUATION_RANGES.items()}

    def collect(name, attribute):
        return np.array([getattr(fit, attribute) for fit in fits[name]]).reshape(energy.shape[:-1])

    return DecayParameters(edt=collect('EDT', 'value'),
                           t20=collect('T20', 'value'),
                           t30=collect('T30', 'value'),
                           xi_t20=collect('T20', 'nonlinearity'),
                           xi_t30=collect('T30', 'nonlinearity'))
