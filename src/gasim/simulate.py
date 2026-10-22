"""One simulation run: config in, Result out."""
import time

from gasim.air import energy_attenuation, speed_of_sound
from gasim.bands import OCTAVE_BANDS
from gasim.config import Config
from gasim.geometry import load_room
from gasim.io.results import Result, run_metadata
from gasim.propagation.rays import trace_rays


def simulate(config: Config):
    """
    Run the new engine on a config (ray tracing with diffuse rain; image sources follow in PR 7)
    :param config: Config
    :return: Result with energy histograms and metadata of the run
    """
    timings = {}
    start = time.perf_counter()
    room, _ = load_room(config)
    timings['load'] = time.perf_counter() - start

    air = config.air
    c = float(speed_of_sound(air.temperature))
    m = energy_attenuation(air.temperature, air.humidity, air.pressure)

    start = time.perf_counter()
    rays = trace_rays(room, config.simulation, c, m)
    timings['rays'] = time.perf_counter() - start

    metadata = run_metadata(config.path, seed=config.simulation.seed, timings=timings, engine='rays',
                            rays=config.simulation.rays, receiver_radius=config.simulation.receiver_radius,
                            escaped_rays=rays.escaped_rays, reflections=rays.reflections,
                            speed_of_sound=c, air_attenuation=m.tolist(), volume=room.volume,
                            surface_area=room.surface_area)
    return Result(energy=rays.energy, time_step=config.simulation.time_step, bands=OCTAVE_BANDS.copy(),
                  source=room.source, receivers=room.receivers, metadata=metadata)
