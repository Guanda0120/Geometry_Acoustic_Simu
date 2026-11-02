"""Room geometry: data model, model building and checks."""
import dataclasses

from gasim.config import Config
from gasim.geometry.build import build_room
from gasim.geometry.room import Room
from gasim.io.rhino import read_3dm
from gasim.surfaces import assign_surfaces


def load_room(config: Config):
    """
    Read the model of a config and build its Room, with the surfaces of the config
    :param config: Config
    :return: Room, list of skipped (layer, geometry type) objects
    """
    model = read_3dm(config.model)
    missing = sorted({layer for layer, _ in model.polygons} - set(config.materials.names))
    if missing:
        raise ValueError(f"No material defined for layer(s): {', '.join(missing)}")
    room = build_room(model.polygons, config.materials, model.source, model.receivers)
    if config.surfaces:
        names, poly_surface = assign_surfaces(room.poly_vertices, room.poly_layer, room.poly_normal,
                                              config.surfaces, config.materials.names)
        room = dataclasses.replace(room, surface_names=names, poly_surface=poly_surface)
    return room, model.skipped


__all__ = ['Room', 'build_room', 'load_room']
