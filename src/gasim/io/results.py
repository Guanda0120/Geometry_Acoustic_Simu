"""
Simulation results: energy histograms per receiver and band, saved as one .npz file.

The file holds plain arrays and a JSON metadata string (config, seed, version, timings), so it can be
read back without pickle and traced to the run that produced it. Since format 2 it can also hold the
image-source paths, for export to Rhino.
"""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from gasim import __version__
from gasim.analysis.decay import decay_parameters
from gasim.propagation.image_sources import ImageSourcePaths

FORMAT_VERSION = 2
READABLE_VERSIONS = (1, 2)
_ARRAYS = ('energy', 'bands', 'source', 'receivers')


@dataclass(frozen=True)
class Result:
    """
    Energy histograms of one run.

    energy[r, b, k] is the energy arriving at receiver r in band b during [k dt, (k + 1) dt), per unit
    area and per unit energy emitted by the source, in 1/m2. For the direct sound alone this is
    exp(-m d) / (4 pi d2).
    """
    energy: np.ndarray  # (R, B, K), 1/m2
    time_step: float  # s
    bands: np.ndarray  # (B,) centre frequencies, Hz
    source: np.ndarray  # (3,)
    receivers: np.ndarray  # (R, 3)
    metadata: dict = field(default_factory=dict)
    paths: ImageSourcePaths = None  # image-source paths, if the run had them

    def __post_init__(self):
        n_receivers, n_bands, _ = self.energy.shape
        if self.bands.shape != (n_bands,) or self.receivers.shape != (n_receivers, 3) or self.source.shape != (3,):
            raise ValueError("Result arrays have inconsistent shapes")

    @property
    def time(self):
        """
        :return: (K,) start time of each bin, s
        """
        return np.arange(self.energy.shape[-1]) * self.time_step

    def decay(self):
        """
        :return: DecayParameters with arrays of shape (R, B)
        """
        return decay_parameters(self.energy, self.time_step)


def run_metadata(config_path=None, seed=None, timings=None, **extra):
    """
    Metadata stored with a result
    :param config_path: Config file of the run; its text is stored too
    :param seed: Random seed of the run
    :param timings: {stage: seconds}
    :param extra: Further JSON-serialisable entries
    :return: dict
    """
    metadata = {'gasim_version': __version__, 'seed': seed, 'timings': timings or {}}
    if config_path is not None:
        metadata['config_path'] = str(config_path)
        metadata['config'] = Path(config_path).read_text()
    metadata.update(extra)
    return metadata


def save_result(result: Result, path):
    """
    Write a result to an .npz file
    :param result: Result
    :param path: Output file; parent folders are created
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(result, name) for name in _ARRAYS}
    if result.paths is not None:
        arrays.update({f'path_{f.name}': getattr(result.paths, f.name) for f in fields(ImageSourcePaths)})
    np.savez_compressed(path, format_version=FORMAT_VERSION, time_step=result.time_step,
                        metadata=json.dumps(result.metadata), **arrays)


def load_result(path):
    """
    Read a result written by save_result
    :param path: .npz file
    :return: Result
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data['format_version'])
        if version not in READABLE_VERSIONS:
            raise ValueError(f"{path}: result format {version}, this gasim reads formats {READABLE_VERSIONS}")
        paths = None
        if 'path_order' in data:
            paths = ImageSourcePaths(**{f.name: data[f'path_{f.name}'] for f in fields(ImageSourcePaths)})
        return Result(time_step=float(data['time_step']), metadata=json.loads(str(data['metadata'])), paths=paths,
                      **{name: data[name] for name in _ARRAYS})
