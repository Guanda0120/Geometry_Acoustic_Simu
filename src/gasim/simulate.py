"""One simulation run: config in, Result out."""
import time

from gasim.air import energy_attenuation, speed_of_sound
from gasim.analysis.surfaces import SurfaceStatistics
from gasim.bands import OCTAVE_BANDS
from gasim.config import Config
from gasim.geometry import load_room
from gasim.io.results import Result, run_metadata
from gasim.propagation.hybrid import hybrid
from gasim.propagation.image_sources import reflections_by_surface


def simulate(config: Config):
    """
    Run the new engine on a config: image sources up to ism_order plus rays with diffuse rain
    :param config: Config
    :return: Result with energy histograms, image-source paths and metadata of the run
    """
    timings = {}
    start = time.perf_counter()
    room, _ = load_room(config)
    timings['load'] = time.perf_counter() - start

    air = config.air
    c = float(speed_of_sound(air.temperature))
    m = energy_attenuation(air.temperature, air.humidity, air.pressure)

    start = time.perf_counter()
    output = hybrid(room, config.simulation, c, m)
    timings['simulation'] = time.perf_counter() - start

    rays = output.rays
    metadata = run_metadata(config.path, seed=config.simulation.seed, timings=timings, engine='hybrid',
                            rays=config.simulation.rays, ism_order=config.simulation.ism_order,
                            image_sources=[output.paths.counts(r).tolist() for r in range(len(room.receivers))],
                            receiver_radius=config.simulation.receiver_radius,
                            escaped_rays=rays.escaped_rays, reflections=rays.reflections,
                            speed_of_sound=c, air_attenuation=m.tolist(), volume=room.volume,
                            surface_area=room.surface_area, layers=list(room.poly_layer),
                            surfaces=list(room.surface_names), poly_surface=room.poly_surface.tolist())
    surfaces = SurfaceStatistics(
        names=room.surface_names, area=room.area_by_surface(),
        ism_reflections=reflections_by_surface(output.paths, room.poly_surface, len(room.receivers), room.n_surfaces),
        ray_hits=rays.surface_hits, incident=rays.surface_incident, absorbed=rays.surface_absorbed,
        received=output.received, direct=output.direct)
    return Result(energy=output.energy, time_step=config.simulation.time_step, bands=OCTAVE_BANDS.copy(),
                  source=room.source, receivers=room.receivers, metadata=metadata, paths=output.paths,
                  surfaces=surfaces)
