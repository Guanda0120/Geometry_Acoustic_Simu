"""
Performance targets of docs/rewrite-plan.md, Sect. 4.6 (8-core laptop). Not run by default:

    uv run pytest -m slow -s
"""
import dataclasses
import time
from pathlib import Path

import pytest

from gasim.config import load_config
from gasim.simulate import simulate

CASES = [  # config, rays, target in seconds
    ('configs/test_room.toml', 31_500, 10),
    ('configs/classroom.toml', 22_200, 60),  # 100 rays per m3, image sources of order 5
    ('configs/mf_hall.toml', 438_900, 600),  # 100 rays per m3
]


@pytest.mark.slow
@pytest.mark.parametrize('config_path, rays, target', CASES, ids=lambda case: Path(str(case)).stem)
def test_meets_the_performance_target(config_path, rays, target):
    config = load_config(config_path)
    if not config.model.exists():
        pytest.skip(f'{config.model} is not available (private model)')
    config = dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, rays=rays))
    simulate(dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, rays=100)))  # compile
    start = time.perf_counter()
    result = simulate(config)
    elapsed = time.perf_counter() - start
    print(f"\n{config_path}: {rays} rays, image sources of order {config.simulation.ism_order}: {elapsed:.1f} s "
          f"(target {target} s)")
    assert result.metadata['escaped_rays'] == 0
    assert elapsed < target
