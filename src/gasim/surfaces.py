"""
Surfaces: named groups of boundary polygons, such as the walls, ceiling and floor of a room.

Materials come from the model layers; surfaces are an independent grouping for the statistics and,
later, the sensitivities of a run. A config defines them in a [surfaces] table, in order:

    [surfaces]
    ceiling = { layers = ["White"] }
    wall_C  = { x = [-0.1, 3.61], y = [2.39, 16.61] }
    wall_D  = { x = [25.59, 30.1], normal = [-1, 0, 0] }
    wall_A  = {}

A polygon belongs to the first group whose conditions it meets: its layer is one of `layers`, all its
vertices lie within the `x`, `y` and `z` ranges, and its inward normal is within `max_angle` degrees
(default 45) of `normal`. A group without conditions takes all remaining polygons; polygons that match
no group form the surface "other". Without a [surfaces] table every layer is one surface.
"""
from dataclasses import dataclass

import numpy as np

OTHER = 'other'
_KEYS = {'layers', 'x', 'y', 'z', 'normal', 'max_angle'}
_TOLERANCE = 1e-6  # m, for the coordinate ranges


@dataclass(frozen=True)
class SurfaceRule:
    """
    Conditions for the polygons of one surface; None means no condition
    """
    name: str
    layers: tuple = None
    bounds: tuple = (None, None, None)  # (min, max) per coordinate x, y, z
    normal: np.ndarray = None  # (3,) unit vector, pointing into the room
    max_angle: float = 45.0  # degrees

    def matches(self, vertices: np.ndarray, layer: str, normal: np.ndarray):
        """
        :param vertices: (n, 3) polygon vertices
        :param layer: Layer of the polygon
        :param normal: (3,) inward unit normal of the polygon
        :return: True if the polygon meets every condition
        """
        if self.layers is not None and layer not in self.layers:
            return False
        for axis, limits in enumerate(self.bounds):
            if limits is not None and (vertices[:, axis].min() < limits[0] - _TOLERANCE
                                       or vertices[:, axis].max() > limits[1] + _TOLERANCE):
                return False
        return self.normal is None or normal @ self.normal >= np.cos(np.radians(self.max_angle))


def _range(value, key: str, name: str):
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or values[0] > values[1]:
        raise ValueError(f"surface {name!r}: {key} must be [min, max], got {value}")
    return float(values[0]), float(values[1])


def surface_rules(spec: dict):
    """
    Build the rules of the [surfaces] table of a config
    :param spec: {name: {'layers': [...], 'x': [min, max], 'y': ..., 'z': ..., 'normal': [...], 'max_angle': deg}}
    :return: Tuple of SurfaceRule in table order
    """
    rules = []
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Surface {name!r} must be a table of conditions")
        if name == OTHER:
            raise ValueError(f"{OTHER!r} is reserved for polygons that match no surface")
        unknown = set(entry) - _KEYS
        if unknown:
            raise ValueError(f"Unknown key(s) for surface {name!r}: {', '.join(sorted(unknown))}")
        layers = entry.get('layers')
        if layers is not None:
            if isinstance(layers, str) or not all(isinstance(layer, str) for layer in layers):
                raise ValueError(f"surface {name!r}: layers must be a list of layer names")
            layers = tuple(layers)
        bounds = tuple(_range(entry[key], key, name) if key in entry else None for key in ('x', 'y', 'z'))
        normal = entry.get('normal')
        if normal is not None:
            normal = np.asarray(normal, dtype=float)
            if normal.shape != (3,) or not np.linalg.norm(normal) > 0:
                raise ValueError(f"surface {name!r}: normal must be a non-zero vector [x, y, z]")
            normal = normal / np.linalg.norm(normal)
        max_angle = float(entry.get('max_angle', 45.0))
        if 'max_angle' in entry and normal is None:
            raise ValueError(f"surface {name!r}: max_angle needs a normal")
        if not 0 <= max_angle <= 180:
            raise ValueError(f"surface {name!r}: max_angle must lie in [0, 180] degrees")
        rules.append(SurfaceRule(name, layers, bounds, normal, max_angle))
    return tuple(rules)


def assign_surfaces(poly_vertices, poly_layer, poly_normal, rules, layer_names):
    """
    Put every polygon on a surface
    :param poly_vertices: P arrays of shape (n_i, 3)
    :param poly_layer: P layer names
    :param poly_normal: (P, 3) inward unit normals
    :param rules: Tuple of SurfaceRule; empty: one surface per layer
    :param layer_names: Layer names in material-table order, the surfaces when there are no rules
    :return: Surface names (tuple), (P,) surface index of each polygon
    """
    if not rules:
        return tuple(layer_names), np.array([layer_names.index(layer) for layer in poly_layer], dtype=np.int64)
    names = [rule.name for rule in rules]
    index = np.full(len(poly_vertices), -1, dtype=np.int64)
    for p, (vertices, layer, normal) in enumerate(zip(poly_vertices, poly_layer, poly_normal)):
        for s, rule in enumerate(rules):
            if rule.matches(np.asarray(vertices), layer, np.asarray(normal)):
                index[p] = s
                break
    if np.any(index < 0):
        names.append(OTHER)
        index[index < 0] = len(names) - 1
    return tuple(names), index
