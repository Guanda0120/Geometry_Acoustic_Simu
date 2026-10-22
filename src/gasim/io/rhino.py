"""
Read room models from Rhinoceros .3dm files.

Every layer holding surfaces is a material; points on the layers 'Source' (exactly one) and
'Receiver' are the source and receivers. The legacy normal-helper layers 'Center_pt' and
'Direc_line' are ignored: normals are now oriented automatically.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rhino3dm

SOURCE_LAYER = 'Source'
RECEIVER_LAYER = 'Receiver'
IGNORED_LAYERS = ('Center_pt', 'Direc_line')
UNIT_SCALE = {rhino3dm.UnitSystem.Meters: 1.0,
              rhino3dm.UnitSystem.Centimeters: 0.01,
              rhino3dm.UnitSystem.Millimeters: 0.001}


@dataclass(frozen=True)
class RhinoModel:
    """
    Contents of a .3dm file in metres
    """
    polygons: list  # (layer name, (n, 3) vertices)
    source: np.ndarray  # (3,)
    receivers: np.ndarray  # (R, 3)
    skipped: list  # (layer name, geometry type) of objects that are not surfaces


def face_polygon(brep: rhino3dm.Brep, face_index: int):
    """
    Ordered corner points of the outer loop of a planar Brep face
    :param brep: Brep
    :param face_index: Face index
    :return: (n, 3) vertices in model units
    """
    face = brep.Faces[face_index]
    if not face.IsPlanar(1e-6):
        raise ValueError("Face is not planar; subdivide curved surfaces into planar polygons")
    if len(face.Loops) != 1:
        raise ValueError("Faces with holes are not supported; split the face around the hole")
    loop = face.OuterLoop
    vertices = []
    for k in range(loop.TrimCount):
        trim = loop.Trims[k]
        if not brep.Edges[trim.EdgeIndex].IsLinear(1e-6):
            raise ValueError("Face has a curved edge; approximate it by straight segments")
        point = brep.Vertices[trim.StartVertexIndex].Location
        vertices.append([point.X, point.Y, point.Z])
    return np.array(vertices)


def read_3dm(path):
    """
    Read polygons, source and receivers from a .3dm file
    :param path: .3dm file
    :return: RhinoModel
    """
    model = rhino3dm.File3dm.Read(str(path))
    if model is None:
        raise FileNotFoundError(f"Cannot read Rhino file {path}")
    scale = UNIT_SCALE.get(model.Settings.ModelUnitSystem)
    if scale is None:
        raise ValueError(f"Unsupported model units {model.Settings.ModelUnitSystem} in {path}")

    polygons, sources, receivers, skipped = [], [], [], []
    for obj in model.Objects:
        layer = model.Layers[obj.Attributes.LayerIndex].Name
        geometry = obj.Geometry
        if layer in IGNORED_LAYERS:
            continue
        if layer in (SOURCE_LAYER, RECEIVER_LAYER):
            if not isinstance(geometry, rhino3dm.Point):
                skipped.append((layer, type(geometry).__name__))
                continue
            location = geometry.Location
            (sources if layer == SOURCE_LAYER else receivers).append([location.X, location.Y, location.Z])
            continue
        if isinstance(geometry, rhino3dm.Extrusion):
            geometry = geometry.ToBrep(True)
        if not isinstance(geometry, rhino3dm.Brep):
            skipped.append((layer, type(geometry).__name__))
            continue
        for face_index in range(len(geometry.Faces)):
            try:
                polygons.append((layer, face_polygon(geometry, face_index) * scale))
            except ValueError as error:
                raise ValueError(f"{Path(path).name}, layer {layer!r}: {error}") from None

    if len(sources) != 1:
        raise ValueError(f"{path} needs exactly one point on layer {SOURCE_LAYER!r}, found {len(sources)}")
    if not receivers:
        raise ValueError(f"{path} has no points on layer {RECEIVER_LAYER!r}")
    return RhinoModel(polygons=polygons,
                      source=np.array(sources[0]) * scale,
                      receivers=np.array(receivers) * scale,
                      skipped=skipped)


def write_paths(paths, path, receiver: int = 0, max_order=None, layers=None):
    """
    Write image-source paths of one receiver as polylines, one layer per reflection order
    :param paths: ImageSourcePaths
    :param path: .3dm file to write
    :param receiver: Receiver index
    :param max_order: Highest order to write, default all
    :param layers: Polygon layer names; if given, each curve is named after the surfaces it reflects from
    :return: Number of paths written
    """
    model = rhino3dm.File3dm()
    model.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Meters
    selected = np.flatnonzero((paths.receiver == receiver)
                              & (paths.order <= (paths.order.max(initial=0) if max_order is None else max_order)))
    layer_of_order = {}
    for index in selected:
        order = int(paths.order[index])
        if order not in layer_of_order:
            name = 'Direct' if order == 0 else f'Order {order}'
            layer_of_order[order] = model.Layers.AddLayer(name, _ORDER_COLOURS[min(order, len(_ORDER_COLOURS) - 1)])
        attributes = rhino3dm.ObjectAttributes()
        attributes.LayerIndex = layer_of_order[order]
        if layers is not None:
            attributes.Name = ' > '.join(layers[p] for p in paths.planes[index, :order]) or 'direct'
        points = [rhino3dm.Point3d(*point) for point in paths.points[index, :order + 2]]
        model.Objects.AddCurve(rhino3dm.PolylineCurve(points), attributes)
    if not model.Write(str(path), 7):
        raise OSError(f"Cannot write {path}")
    return len(selected)


_ORDER_COLOURS = [(0, 0, 0, 255), (220, 40, 40, 255), (240, 150, 20, 255), (40, 150, 60, 255),
                  (30, 100, 220, 255), (140, 60, 200, 255)]
