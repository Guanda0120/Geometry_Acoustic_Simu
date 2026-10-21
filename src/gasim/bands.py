"""Frequency bands of the new engine."""
import numpy as np

# Octave-band centre frequencies, Hz
OCTAVE_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
N_BANDS = OCTAVE_BANDS.size
