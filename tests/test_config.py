from pathlib import Path

import numpy as np
import pytest

from gasim.config import load_config
from gasim.materials import material_table

CONFIGS = sorted(Path('configs').glob('*.toml'))


def write(tmp_path, text):
    path = tmp_path / 'config.toml'
    path.write_text(text)
    return path


MINIMAL = '''
[room]
model = "models/Simple_Test.3dm"
[materials]
Wood = { absorption = 0.1 }
'''


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_bundled_configs_load(path):
    config = load_config(path)
    assert config.materials.absorption.shape == (len(config.materials.names), 6)
    assert config.simulation.rays > 0


def test_defaults_and_broadcasting(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    np.testing.assert_array_equal(config.materials.absorption, np.full((1, 6), 0.1))
    np.testing.assert_array_equal(config.materials.scattering, np.zeros((1, 6)))
    assert config.simulation.seed == 1
    assert config.air.temperature == 20.0


@pytest.mark.parametrize('extra, message', [
    ('[simulaton]\nrays = 10\n', 'Unknown section'),
    ('[simulation]\nray = 10\n', 'Unknown key'),
    ('[simulation]\nrays = 0\n', 'rays'),
    ('[simulation]\nmax_time = 0.0001\n', 'max_time'),
    ('[air]\nhumidity = 120\n', 'humidity'),
])
def test_invalid_settings_are_rejected(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, MINIMAL + extra))


@pytest.mark.parametrize('entry, message', [
    ({'absorption': 1.2}, r'\[0, 1\]'),
    ({'absorption': [0.1, 0.2]}, '1 or 6 values'),
    ({'scattering': 0.1}, 'no absorption'),
    ({'absorption': 0.1, 'colour': 'red'}, 'Unknown key'),
])
def test_invalid_materials_are_rejected(entry, message):
    with pytest.raises(ValueError, match=message):
        material_table({'Wall': entry})


def test_unknown_material_name():
    table = material_table({'Wall': {'absorption': 0.1}})
    with pytest.raises(KeyError, match='Floor'):
        table.index('Floor')
