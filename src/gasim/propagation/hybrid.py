"""
Hybrid of image sources and ray tracing (docs/rewrite-plan.md, Sect. 4.4).

The image sources give the direct sound and every purely specular path up to ism_order exactly; the
rays skip those paths (order exclusion) and carry everything else: higher orders and every path that
scatters at least once, including the diffuse rain.
"""
from dataclasses import dataclass

from gasim.config import SimulationSettings
from gasim.geometry.room import Room
from gasim.propagation.image_sources import (
    ImageSourcePaths,
    add_to_histogram,
    find_image_sources,
    received_by_last_surface,
)
from gasim.propagation.rays import RayTracing, trace_rays


@dataclass(frozen=True)
class HybridResult:
    energy: object  # (R, B, K) histogram, 1/m2 per unit source energy
    paths: ImageSourcePaths
    rays: RayTracing
    received: object  # (R, B, S) energy per receiver by the surface of the last reflection, 1/m2
    direct: object  # (R, B) direct sound, 1/m2; received.sum(-1) + direct = energy.sum(-1)


def hybrid(room: Room, settings: SimulationSettings, speed_of_sound: float, air_attenuation):
    """
    Energy histograms from image sources up to settings.ism_order plus rays for the rest
    :param room: Room
    :param settings: SimulationSettings
    :param speed_of_sound: m/s
    :param air_attenuation: (B,) intensity attenuation constant m per band, 1/m
    :return: HybridResult
    """
    paths = find_image_sources(room, settings.ism_order, air_attenuation)
    rays = trace_rays(room, settings, speed_of_sound, air_attenuation, exclude_specular_order=settings.ism_order)
    energy = rays.energy.copy()
    add_to_histogram(paths, energy, settings.time_step, speed_of_sound)
    received, direct = received_by_last_surface(paths, room.poly_surface, len(room.receivers), room.n_surfaces,
                                                energy.shape[2], settings.time_step, speed_of_sound)
    return HybridResult(energy, paths, rays, received + rays.surface_received, direct)
