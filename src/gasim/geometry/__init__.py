"""Room geometry: data model, model building and checks."""
from gasim.config import Config
from gasim.geometry.build import build_room
from gasim.geometry.room import Room
from gasim.io.rhino import read_3dm


def load_room(config: Config):
    """
    Read the model of a config and build its Room
    :param config: Config
    :return: Room, list of skipped (layer, geometry type) objects
    """
    model = read_3dm(config.model)
    missing = sorted({layer for layer, _ in model.polygons} - set(config.materials.names))
    if missing:
        raise ValueError(f"No material defined for layer(s): {', '.join(missing)}")
    return build_room(model.polygons, config.materials, model.source, model.receivers), model.skipped


__all__ = ['Room', 'build_room', 'load_room']
