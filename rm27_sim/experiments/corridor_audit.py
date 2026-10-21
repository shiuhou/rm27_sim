"""Conservative swept-sphere audit against colliders AND visual convex hulls.

No physics is advanced. Clear means only the declared static geometry envelope
is clear; an obstruction against a visual hull can be a conservative false hit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh


def mesh_surface_hits(model, data, geom, points, radius):
    """Exact distance to mesh triangles after a conservative per-triangle AABB filter.

    Measures a surface, not solid containment, for visual-only meshes.
    """
    mesh = model.geom_dataid[geom]
    va, fa = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
    vertices = model.mesh_vert[va:va+model.mesh_vertnum[mesh]]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    faces = model.mesh_face[fa:fa+model.mesh_facenum[mesh]]
    triangles = vertices[faces]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    result = []
    for index, p in enumerate(points):
        delta = np.maximum(np.maximum(low-p, p-high), 0)
        candidates = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= radius**2)
        if not len(candidates):
            continue
        nearest = trimesh.triangles.closest_point(triangles[candidates], np.broadcast_to(p, (len(candidates), 3)))
        distances = np.linalg.norm(nearest-p, axis=1)
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite mesh surface distance; cannot certify clearance')
        j = int(np.argmin(distances))
        if distances[j] <= radius:
            result.append(dict(sample=index, gap_m=float(distances[j]-radius),
                               triangle=int(candidates[j]), closest_point_world_m=nearest[j].tolist()))
    return result


def path_samples(waypoints, spacing):
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected at least two finite world waypoints')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive')
    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(b-a)/spacing)))
        result.extend(a+(b-a)*i/steps for i in range(1, steps+1))
    return np.asarray(result)


def audit(scene, waypoints, output, spacing=.02, tracking_margin=.08):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    points = path_samples(waypoints, spacing)
    if not np.isfinite(tracking_margin) or tracking_margin < 0:
        raise ValueError('tracking margin must be finite and nonnegative')
    root = ET.parse(scene).getroot()
    original = mujoco.MjModel.from_xml_path(str(scene))
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(original, original_data)
    body = original.body('drone_1').id
    vehicle_geoms = np.flatnonzero(original.geom_bodyid == body)
    # Each geom's compiled bounding sphere plus its offset encloses the vehicle
    # at every attitude; conservative, not the nominal horizontal footprint.
    bounds = []
    for g in vehicle_geoms:
        offset = original_data.geom_xpos[g]-original_data.xpos[body]
        if original.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = original_data.geom_xmat[g].reshape(3, 3)[:, 2]
            half = original.geom_size[g, 1]
            bounds.append(max(np.linalg.norm(offset-axis*half), np.linalg.norm(offset+axis*half))
                          + original.geom_size[g, 0])
        else:
            bounds.append(np.linalg.norm(offset)+original.geom_rbound[g])
    radius = float(max(bounds))
    inflation = spacing/2 + tracking_margin
    probe = ET.SubElement(root.find('worldbody'), 'body', name='corridor_probe')
    ET.SubElement(probe, 'freejoint', name='corridor_probe_joint')
    ET.SubElement(probe, 'geom', name='corridor_probe_geom', type='sphere',
                  size=str(radius+inflation), mass='1', contype='0', conaffinity='0')
    # Generated RM27 scenes have absolute paths. Resolve relative assets too.
    compiler = root.find('compiler')
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            directory = '' if compiler is None else compiler.get(
                'meshdir' if asset.tag == 'mesh' else 'texturedir', compiler.get('assetdir', ''))
            asset.set('file', str((scene.parent/directory/asset.get('file')).resolve()))
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(key, None)
    asset_hashes = {asset.get('file'): hashlib.sha256(Path(asset.get('file')).read_bytes()).hexdigest()
                    for asset in root.findall('asset/*') if asset.get('file')}
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    probe_id = model.geom('corridor_probe_geom').id
    qa = model.jnt_qposadr[model.joint('corridor_probe_joint').id]
    selected_body = model.body('drone_1').id
    others = [g for g in range(model.ngeom) if g != probe_id and model.geom_bodyid[g] != selected_body]
    supported = {int(x) for x in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE,
                 mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                 mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH)}
    if any(int(model.geom_type[g]) not in supported for g in others):
        raise ValueError('Unsupported corridor geometry type')
    hits = {}
    for sample_index, position in enumerate(points):
        data.qpos[qa:qa+3] = position
        mujoco.mj_forward(model, data)
        for g in others:
            # Plane bounds are infinite even though geom_rbound is zero.
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and np.linalg.norm(data.geom_xpos[g]-position) > model.geom_rbound[g]+radius+inflation:
                continue
            distance = mujoco.mj_geomDistance(model, data, probe_id, g, .001, None)
            if not np.isfinite(distance):
                raise ValueError('Nonfinite geometry distance')
            if distance <= 0:
                name = model.geom(g).name
                row = hits.setdefault(name, dict(samples=0, first_sample=sample_index,
                    first_position_world_m=position.tolist(), minimum_signed_gap_m=float(distance),
                    collision_enabled=bool(model.geom_contype[g] or model.geom_conaffinity[g]),
                    geom_type=int(model.geom_type[g])))
                row['samples'] += 1
                row['minimum_signed_gap_m'] = min(row['minimum_signed_gap_m'], float(distance))
    surface_checks = {}
    for name, row in hits.items():
        if not row['collision_enabled'] and row['geom_type'] == int(mujoco.mjtGeom.mjGEOM_MESH):
            exact = mesh_surface_hits(model, data, model.geom(name).id, points, radius+inflation)
            surface_checks[name] = {'hull_only_obstruction': not exact, 'surface_hits': exact}
    actual_obstructions = [name for name, row in hits.items()
                           if row['collision_enabled'] or name not in surface_checks
                           or surface_checks[name]['surface_hits']]
    result = dict(passed=not actual_obstructions,
                  convex_envelope_clear=not hits,
                  scope='static collider and visual-surface sweep only; no solid-containment or flight certification',
                  scene=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                  asset_sha256=asset_hashes,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  mujoco_version=mujoco.__version__,
                  waypoints_world_m=np.asarray(waypoints).tolist(), sample_count=len(points),
                  spacing_m=spacing, vehicle_enclosing_radius_m=radius, sampling_inflation_m=spacing/2,
                  total_inflation_m=inflation,
                  tracking_margin_m=tracking_margin,
                  obstructions=hits, s1_complete=False,
                  visual_surface_checks=surface_checks,
                  surface_route_clear=not actual_obstructions,
                  remaining_surface_or_collider_obstructions=actual_obstructions,
                  limitations='Includes visual-only convex hulls; hits may be conservative. Clear result excludes takeoff, docking, contact, control tracking error and unmodelled geometry.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--path', type=Path, required=True, help='JSON list of world XYZ waypoints')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracking-margin', type=float, default=.08)
    args = parser.parse_args()
    result = audit(args.scene, json.loads(args.path.read_text()), args.output,
                   tracking_margin=args.tracking_margin)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
