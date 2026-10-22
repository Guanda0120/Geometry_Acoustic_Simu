import numpy as np
import pytest

from gasim import __version__
from gasim.bands import OCTAVE_BANDS
from gasim.io.results import Result, load_result, run_metadata, save_result


def make_result(metadata=None):
    rng = np.random.default_rng(3)
    k = np.arange(3000)
    decay = np.exp(-np.log(1e6) * k * 1e-3 / np.linspace(0.6, 1.1, 6)[:, None])
    energy = np.stack([decay * rng.uniform(0.5, 1.5) for _ in range(2)])
    return Result(energy=energy, time_step=1e-3, bands=OCTAVE_BANDS.copy(), source=np.array([1.0, 2.0, 1.5]),
                  receivers=np.array([[3.0, 4.0, 1.2], [5.0, 2.0, 1.2]]), metadata=metadata or {})


def test_round_trip(tmp_path):
    result = make_result(run_metadata('configs/test_room.toml', seed=7, timings={'rays': 1.5}))
    path = tmp_path / 'nested' / 'run.npz'
    save_result(result, path)
    loaded = load_result(path)

    for name in ('energy', 'bands', 'source', 'receivers'):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    assert loaded.time_step == result.time_step
    assert loaded.metadata == result.metadata
    assert loaded.metadata['gasim_version'] == __version__
    assert '[room]' in loaded.metadata['config']


def test_decay_of_a_result():
    parameters = make_result().decay()
    assert parameters.t30.shape == (2, 6)
    np.testing.assert_allclose(parameters.t30[1], np.linspace(0.6, 1.1, 6), rtol=1e-6)


def test_inconsistent_shapes_are_rejected():
    result = make_result()
    with pytest.raises(ValueError, match='inconsistent'):
        Result(energy=result.energy, time_step=1e-3, bands=OCTAVE_BANDS[:5], source=result.source,
               receivers=result.receivers)


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / 'run.npz'
    save_result(make_result(), path)
    with np.load(path) as data:
        arrays = dict(data)
    arrays['format_version'] = 99
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='format 99'):
        load_result(path)
