from pathlib import Path

import numpy as np
import pytest
import rhino3dm

from gasim.cli import main
from gasim.geometry.build import build_room
from gasim.geometry.checks import check_room
from gasim.io.rhino import read_3dm
from gasim.materials import material_table

MODELS = sorted(Path('models').glob('*.3dm'))


def write_box_model(path, units=rhino3dm.UnitSystem.Millimeters, scale=1000.0):
    """A 2 x 3 x 4 m box drawn as one six-face Brep, with source and receiver points"""
    model = rhino3dm.File3dm()
    model.Settings.ModelUnitSystem = units

    def on_layer(name):
        attributes = rhino3dm.ObjectAttributes()
        attributes.LayerIndex = model.Layers.AddLayer(name, (128, 128, 128, 255))
        return attributes

    box = rhino3dm.Box(rhino3dm.BoundingBox(rhino3dm.Point3d(0, 0, 0), rhino3dm.Point3d(2 * scale, 3 * scale, 4 * scale)))
    model.Objects.AddBrep(rhino3dm.Brep.CreateFromBox(box), on_layer('Concrete'))
    model.Objects.AddPoint(rhino3dm.Point3d(1 * scale, 1 * scale, 1 * scale), on_layer('Source'))
    receiver = on_layer('Receiver')
    for y in (1.5, 2.5):
        model.Objects.AddPoint(rhino3dm.Point3d(1 * scale, y * scale, 1.2 * scale), receiver)
    assert model.Write(str(path), 7)
    return path


def test_multi_face_brep_in_millimetres(tmp_path):
    model = read_3dm(write_box_model(tmp_path / 'box.3dm'))
    assert len(model.polygons) == 6
    np.testing.assert_allclose(model.source, [1, 1, 1])
    assert model.receivers.shape == (2, 3)
    room = build_room(model.polygons, material_table({'Concrete': {'absorption': 0.02}}), model.source,
                      model.receivers)
    assert room.volume == pytest.approx(24.0)


def test_test_room_model():
    model = read_3dm('models/Simple_Test.3dm')
    assert len(model.polygons) == 6
    assert model.receivers.shape == (4, 3)
    assert model.skipped == []


@pytest.mark.parametrize('path', MODELS, ids=lambda p: p.stem)
def test_bundled_models_are_closed(path):
    model = read_3dm(path)
    materials = material_table({layer: {'absorption': 0.1} for layer, _ in model.polygons})
    room = build_room(model.polygons, materials, model.source, model.receivers)
    assert room.volume > 0
    assert check_room(room, n_rays=500).ok


def test_cli_check(capsys):
    assert main(['check', 'configs/test_room.toml']) == 0
    output = capsys.readouterr().out
    assert 'Volume     315.0 m3' in output
    assert output.strip().endswith('OK')
