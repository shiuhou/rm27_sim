"""Build finite convex surface shells from a field's non-colliding visual meshes.

This follows the old field's surface-partition approach. It never takes a convex
hull of an entire object: holes and disconnected patches are retained. Normals
are clustered to reduce CAD tessellation, but each accepted plane is checked
against every source vertex. Small disconnected surface details are omitted
according to an explicit, recorded area threshold.

Run with ``python -m rm27_sim.simulation.field_collision MODEL OUTPUT``.
The output contains collision includes only; it does not modify the input model.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mapbox_earcut
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
import shapely
from shapely.geometry import Polygon


def _numbers(values):
    return " ".join(format(float(x), ".10g") for x in np.asarray(values).ravel())


def convex_parts(polygon: Polygon):
    """Partition a polygon, including holes, and merge only convex neighbours."""
    if not polygon.interiors and polygon.convex_hull.area - polygon.area < 1e-10:
        return [polygon]
    # CAD polygons can have holes touching an outer vertex. Earcut handles
    # these valid weakly-simple rings without filling or buffering their holes.
    rings = [np.asarray(r.coords)[:-1] for r in [polygon.exterior, *polygon.interiors]]
    coordinates = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    indices = mapbox_earcut.triangulate_float64(coordinates, ends).reshape(-1, 3)
    triangles = list(shapely.polygons(coordinates[indices]))
    if abs(sum(p.area for p in triangles)-polygon.area) > max(1e-10, polygon.area*1e-8):
        raise ValueError('surface triangulation did not preserve polygon area')
    parts = dict(enumerate(triangles))
    owners = {}
    queue = deque()

    def edges(poly):
        points = np.round(np.asarray(poly.exterior.coords)[:-1], 8)
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            yield tuple(sorted((tuple(a), tuple(b))))

    def add(index):
        for edge in edges(parts[index]):
            ids = owners.setdefault(edge, set())
            ids.add(index)
            if len(ids) == 2:
                queue.append(tuple(ids))

    def remove(index):
        for edge in edges(parts[index]):
            ids = owners[edge]
            ids.discard(index)
            if not ids:
                del owners[edge]
        del parts[index]

    for i in parts:
        add(i)
    serial = len(parts)
    while queue:
        a, b = queue.popleft()
        if a not in parts or b not in parts:
            continue
        joined = parts[a].union(parts[b])
        if not isinstance(joined, Polygon) or joined.interiors:
            continue
        hull = joined.convex_hull
        if hull.area - joined.area > max(1e-10, joined.area * 1e-8):
            continue
        if len(hull.exterior.coords) > 33:
            continue
        remove(a)
        remove(b)
        # Keep collinear boundary vertices while merging: neighbouring triangles
        # may share only a sub-edge. Removing them here prevents later merges.
        parts[serial] = joined
        add(serial)
        serial += 1
    return [part.convex_hull for part in parts.values()]


def _surface_labels(vertices, faces):
    # Weld coincident CAD seam vertices without closing macroscopic holes.
    _, inverse = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    f = inverse[faces]
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                       shape=(int(inverse.max()) + 1,) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels[f[:, 0]]


def _plane_groups(triangles, normals, areas, labels, tolerance):
    """Yield nearly planar groups, with a measured maximum displacement."""
    center = np.zeros((int(labels.max()) + 1, 3))
    for axis in range(3):
        center[:, axis] = np.bincount(labels, weights=triangles[:, :, axis].mean(axis=1))
    center /= np.bincount(labels)[:, None]
    offsets = np.einsum('ij,ij->i', normals, triangles[:, 0] - center[labels])

    def split(indices, angular_bin, offset_bin, depth):
        normal = np.sum(normals[indices] * areas[indices, None], axis=0)
        normal /= np.linalg.norm(normal)
        offset = float(np.average(triangles[indices].mean(axis=1) @ normal,
                                  weights=areas[indices]))
        error = float(np.max(np.abs(triangles[indices] @ normal - offset)))
        if error <= tolerance:
            yield indices, normal, offset, error
            return
        keys = np.column_stack([np.round(normals[indices] / angular_bin),
                                np.round(offsets[indices] / offset_bin)])
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        if depth >= 10 or (len(indices) == 1):
            for i in indices:
                yield np.array([i]), normals[i], float(normals[i] @ triangles[i, 0]), 0.
            return
        order = np.argsort(inv, kind='stable')
        for part in np.split(order, np.flatnonzero(np.diff(inv[order])) + 1):
            yield from split(indices[part], angular_bin / 2, offset_bin / 2, depth + 1)

    keys = np.column_stack([labels, np.round(normals / .05), np.round(offsets / .005)])
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    for group in np.split(order, np.flatnonzero(np.diff(inv[order])) + 1):
        yield from split(group, .025, .0025, 0)


def surface_parts(vertices, faces, *, minimum_area, plane_tolerance,
                  minimum_hole_width=.01, outline_tolerance=.002):
    """Yield convex polygons in 3-D, with the source plane displacement."""
    labels = _surface_labels(vertices, faces)
    triangles = vertices[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    valid = lengths > 1e-12
    triangles, labels, raw, lengths = triangles[valid], labels[valid], raw[valid], lengths[valid]
    if not len(triangles):
        return
    normals = raw / lengths[:, None]
    signs = np.sign(normals[np.arange(len(normals)), np.argmax(abs(normals), axis=1)])
    normals *= signs[:, None]
    areas = lengths / 2
    for indices, normal, offset, error in _plane_groups(triangles, normals, areas, labels, plane_tolerance):
        if areas[indices].sum() < minimum_area:
            continue
        axis = np.eye(3)[np.argmin(abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        t = triangles[indices]
        projected = np.stack([t @ u, t @ v], axis=-1)
        # Snap each triangle before union so CAD slivers that collapse on the
        # precision grid are removed instead of creating invalid overlay edges.
        polygons_2d = shapely.set_precision(shapely.make_valid(shapely.polygons(projected)), 1e-6)
        patches = shapely.union_all(polygons_2d)
        polygons = [patches] if isinstance(patches, Polygon) else list(patches.geoms)
        for polygon in polygons:
            if not isinstance(polygon, Polygon) or polygon.area < minimum_area:
                continue
            # Holes that cannot contain a 10 mm disk are fastener-scale detail.
            # Filling only those holes adds material at most 5 mm from an edge;
            # usable openings and concave exterior outlines remain partitioned.
            holes = [ring.coords for ring in polygon.interiors
                     if minimum_hole_width == 0 or
                     not Polygon(ring).buffer(-minimum_hole_width/2).is_empty]
            polygon = Polygon(polygon.exterior.coords, holes)
            polygon = polygon.simplify(outline_tolerance, preserve_topology=True)
            for part in convex_parts(polygon):
                points = np.asarray(part.exterior.coords)[:-1]
                yield points[:, :1] * u + points[:, 1:] * v + offset * normal, normal, error


def _add_shell(assets, body, cache, points, normal, name, thickness):
    """Use tight oriented boxes or shared, pose-independent convex meshes."""
    center = points.mean(axis=0)
    edges = np.roll(points, -1, axis=0) - points
    u = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    xy = np.column_stack([(points-center) @ u, (points-center) @ v])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    corners = np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    rectangle = (len(points) == 4 and
                 np.max(shapely.distance(shapely.points(corners), Polygon(xy))) <= .0001)
    # Thin shells need their own contact stiffness; equal-priority mixing
    # with a dynamic geom's default .02 s time constant can allow tunnelling.
    common = {'name': name, 'class': 'collision', 'mass': '0',
              'priority': '1', 'solref': '.004 1'}
    if rectangle:
        center = center + (lo[0]+hi[0])/2*u + (lo[1]+hi[1])/2*v
        rotation = Rotation.from_matrix(np.column_stack([u,v,normal])).as_quat()
        ET.SubElement(body, 'geom', type='box', pos=_numbers(center),
                      quat=_numbers(rotation[[3,0,1,2]]),
                      size=_numbers([*(hi-lo)/2,thickness/2]), **common)
        return
    variants = []
    for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        local = xy * [sx,sy]
        rounded = np.round(local,5)
        if len(np.unique(rounded,axis=0)) < 3:
            rounded = np.round(local,10)
        key = tuple(sorted(map(tuple, rounded)))
        variants.append((key,sx,sy,local))
    key,sx,sy,local = min(variants,key=lambda item:item[0])
    if key not in cache:
        mesh_name = f'rmuc2027_surface_mesh_{len(cache):05d}'
        cache[key] = mesh_name
        prism = np.vstack([np.column_stack([local,np.full(len(local),thickness/2)]),
                           np.column_stack([local,np.full(len(local),-thickness/2)])])
        ET.SubElement(assets,'mesh',name=mesh_name,vertex=_numbers(prism))
    rotation = Rotation.from_matrix(np.column_stack([u*sx,v*sy,normal*sx*sy])).as_quat()
    ET.SubElement(body,'geom',type='mesh',mesh=cache[key],pos=_numbers(center),
                  quat=_numbers(rotation[[3,0,1,2]]),**common)


def build(model_path: Path, output: Path, *, minimum_area=.001,
          plane_tolerance=.001, thickness=.005, minimum_hole_width=.01,
          outline_tolerance=.002):
    if minimum_area <= 0 or plane_tolerance <= 0 or thickness <= 2 * plane_tolerance:
        raise ValueError('positive tolerances and thickness > 2 * plane_tolerance required')
    output.mkdir(parents=True, exist_ok=False)
    model_path = model_path.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assets_root = ET.Element('mujocoinclude')
    assets = ET.SubElement(assets_root, 'asset')
    geoms_root = ET.Element('mujocoinclude')
    body = ET.SubElement(geoms_root, 'body', name='rmuc2027_collision_surfaces')
    records = []
    seen = set()
    mesh_cache = {}
    count = 0
    start = time.monotonic()
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name.startswith('visual_') or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]].astype(float)
        vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]]
        source = model_path.parent / 'assets' / (name + '.msh')
        record = {'source': name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'source_triangles': len(faces), 'parts': 0, 'max_plane_displacement_m': 0.}
        for points, normal, error in surface_parts(vertices, faces, minimum_area=minimum_area,
                                                   plane_tolerance=plane_tolerance,
                                                   minimum_hole_width=minimum_hole_width,
                                                   outline_tolerance=outline_tolerance):
            signature = tuple(sorted(map(tuple, np.round(points, 5))))
            if signature in seen:
                continue
            seen.add(signature)
            mesh_name = f'rmuc2027_surface_{count:05d}'
            _add_shell(assets,body,mesh_cache,points,normal,mesh_name,thickness)
            record['parts'] += 1
            record['max_plane_displacement_m'] = max(record['max_plane_displacement_m'], error)
            count += 1
        records.append(record)
        print(f'{name}: {record["parts"]} parts; cumulative {count}', flush=True)
    for filename, tree in [('collision_assets.xml', assets_root), ('collision_geoms.xml', geoms_root)]:
        ET.indent(tree, space='  ')
        ET.ElementTree(tree).write(output/filename, encoding='UTF-8', xml_declaration=True)
    manifest = {'method': 'convex partitions of bounded-error surface planes; finite centered shells',
                'minimum_patch_area_m2': minimum_area, 'maximum_plane_displacement_m': plane_tolerance,
                'shell_thickness_m': thickness, 'outline_simplification_m': outline_tolerance,
                'minimum_hole_width_m': minimum_hole_width, 'collision_geoms': count,
                'unique_collision_meshes': len(mesh_cache), 'mesh_sharing_grid_m': 1e-5,
                'box_fit_tolerance_m': .0001, 'contact_priority': 1,
                'contact_solref': [.004,1],
                'elapsed_seconds': time.monotonic()-start, 'sources': records,
                'outputs': {p: hashlib.sha256((output/p).read_bytes()).hexdigest()
                            for p in ['collision_assets.xml','collision_geoms.xml']}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--minimum-area', type=float, default=.001)
    parser.add_argument('--plane-tolerance', type=float, default=.001)
    parser.add_argument('--thickness', type=float, default=.005)
    parser.add_argument('--minimum-hole-width', type=float, default=.01)
    parser.add_argument('--outline-tolerance', type=float, default=.002)
    args = parser.parse_args()
    result = build(args.model, args.output, minimum_area=args.minimum_area,
                   plane_tolerance=args.plane_tolerance, thickness=args.thickness,
                   minimum_hole_width=args.minimum_hole_width,
                   outline_tolerance=args.outline_tolerance)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}, indent=2))


if __name__ == '__main__':
    main()
