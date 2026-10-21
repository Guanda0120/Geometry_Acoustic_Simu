"""Characterisation tests of the thesis-2023 engine: they pin down what it does, not what is correct."""
import numpy as np
import pytest

from gasim.cli import main
from gasim.legacy.simulation import load_config, simulation, resolve_path
from gasim.legacy.Acoustic.Eyring import eyring
from gasim.legacy.Acoustic.MaterialDict import MaterialDict
from gasim.legacy.FileIO.ReadRhino import Read3dm
from gasim.legacy.RoomGeometry.FieldGeometry import FieldGeometry

TEST_ROOM = 'configs/legacy/test_room.py'


def test_eyring_test_room():
    config = load_config(TEST_ROOM)
    material_dict = MaterialDict(config['absorptionDict'], config['scatterDict'],
                                 opt_max_res=config['opt_max_res'], model_name=config['model_name'])
    rh_plane, _, _, correct_line = Read3dm(resolve_path(config['fileDir'])).convert_file()
    sound_field = FieldGeometry(rh_plane, material_dict, correct_line)

    e_rt, s_rt, _ = eyring(sound_field, 9 * 7 * 5)

    np.testing.assert_allclose(e_rt, [1.855, 1.015, 1.242, 0.668, 0.497, 0.403], atol=2e-3)
    assert np.all(s_rt > e_rt)


def test_small_run_writes_results(tmp_path):
    config = load_config(TEST_ROOM)
    config.update(num_ray=20, im_order=1, end_rt_spl=30, output_folder=str(tmp_path))

    receiver_group = simulation(config)

    for rec in receiver_group.receiver_container:
        assert rec.rt_time_bi.shape == (6,)
        assert np.all(np.isfinite(rec.rt_time_bi))
    assert (tmp_path / 'rec_0.pkl').exists()
    assert (tmp_path / 'rec_0_res.wav').exists()


def test_cli_rejects_unknown_engine():
    with pytest.raises(SystemExit):
        main(['run', TEST_ROOM, '--engine', 'unknown'])
