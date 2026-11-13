"""Build a source-pinned, articulated blue-base carriage from the static field."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from rm27_sim.paths import ROOT, resolve_asset
from rm27_sim.simulation.field_collision import _surface_labels


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    return ' '.join(format(float(x), '.12g') for x in np.asarray(values).ravel())


def read_mesh(path):
    with Path(path).open('rb') as f:
        nv, nn, nt, nf = np.fromfile(f, '<i4', 4)
        vertices = np.fromfile(f, '<f4', int(nv)*3).reshape(-1, 3)
        normals = np.fromfile(f, '<f4', int(nn)*3).reshape(-1, 3)
        uv = np.fromfile(f, '<f4', int(nt)*2).reshape(-1, 2)
        faces = np.fromfile(f, '<i4', int(nf)*3).reshape(-1, 3)
    if len(normals) not in (0, len(vertices)) or len(uv) not in (0, len(vertices)):
        raise ValueError('Mesh attributes must be indexed by vertex to split losslessly')
    return vertices, normals, uv, faces


def write_subset(path, mesh, mask):
    vertices, normals, uv, faces = mesh
    ids, inverse = np.unique(faces[mask], return_inverse=True)
    normals = normals[ids] if len(normals) else normals
    uv = uv[ids] if len(uv) else uv
    with Path(path).open('wb') as f:
        f.write(struct.pack('<4i', len(ids), len(normals), len(uv), int(mask.sum())))
        vertices[ids].astype('<f4').tofile(f)
        normals.astype('<f4').tofile(f)
        uv.astype('<f4').tofile(f)
        inverse.astype('<i4').ravel().tofile(f)


def travel_range(guide_points, moving_points, axis):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    rail = np.asarray(guide_points) @ axis
    carriage = np.asarray(moving_points) @ axis
    limits = np.array([rail.min()-carriage.min(), rail.max()-carriage.max()])
    if not (limits[0] < 0 < limits[1]):
        raise ValueError('The centered carriage must fit inside the guide endpoints')
    return limits, [float(rail.min()), float(rail.max())], [float(carriage.min()), float(carriage.max())]


def runtime_hashes(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('*.xml')) + sorted((directory/'assets').rglob('*'))
    return {str(p.relative_to(directory)): sha256(p) for p in paths if p.is_file()}


def build(output: Path, config: Path = ROOT/'configs/base_rail.json', *, compile_model=True):
    cfg = json.loads(Path(config).read_text())
    source = resolve_asset(cfg['field_path']).resolve().parent
    for name, expected in cfg['source_files'].items():
        if sha256(source/name) != expected:
            raise ValueError(f'{name} changed; review carriage selection before rebuilding')
    meshes = {}
    labels = {}
    moving_points = []
    for name, rule in cfg['sources'].items():
        path = source/'assets'/f'{name}.msh'
        if sha256(path) != rule['sha256']:
            raise ValueError(f'{name} changed; review connected-component selection')
        mesh = meshes[name] = read_mesh(path)
        vertices, _, _, faces = mesh
        lab = labels[name] = _surface_labels(vertices, faces)
        if set(rule['moving_components']) - set(lab):
            raise ValueError(f'{name}: missing moving components')
        moving_points.append(vertices[np.unique(faces[np.isin(lab, rule['moving_components'])])])
    guide = cfg['guide']
    guide_mesh = meshes[guide['source']]
    guide_faces = guide_mesh[3][np.isin(labels[guide['source']], guide['components'])]
    if not len(guide_faces):
        raise ValueError('No guide geometry selected')
    limits, rail_span, moving_span = travel_range(
        guide_mesh[0][np.unique(guide_faces)], np.vstack(moving_points), cfg['axis'])

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'assets', output/'assets')
    for name in ['model.xml', 'visual_assets.xml', 'visual_geoms.xml', 'collision_assets.xml', 'collision_geoms.xml']:
        shutil.copyfile(source/name, output/name)
    shutil.copyfile(source/'provenance.json', output/'source_provenance.json')
    model = ET.parse(output/'model.xml')
    visuals = ET.parse(output/'visual_geoms.xml')
    assets = ET.parse(output/'visual_assets.xml')
    collisions = ET.parse(output/'collision_geoms.xml')
    world = model.find('worldbody')
    support = ET.SubElement(world, 'body', name='blue_rail_support')
    pivot = np.asarray(cfg['pivot_m'], float)
    carriage = ET.SubElement(world, 'body', name='blue_rail_carriage', pos=numbers(pivot))
    dynamics = cfg['dynamics_estimate']
    ET.SubElement(carriage, 'inertial', pos='0 0 0', mass=str(dynamics['mass_kg']),
                  diaginertia=numbers(dynamics['diaginertia_kg_m2']))
    ET.SubElement(carriage, 'joint', name='blue_base_slide', type='slide',
                  axis=numbers(cfg['axis']), limited='true', range=numbers(limits), damping='1')
    actuator = ET.SubElement(model.getroot(), 'actuator')
    force = dynamics['force_limit_n']
    ET.SubElement(actuator, 'position', name='blue_base_drive', joint='blue_base_slide',
                  kp=str(dynamics['kp']), kv=str(dynamics['kv']), ctrllimited='true',
                  ctrlrange=numbers(limits), forcelimited='true', forcerange=numbers([-force, force]))
    contact = ET.SubElement(model.getroot(), 'contact')
    ET.SubElement(contact, 'exclude', body1='blue_rail_carriage', body2='blue_rail_support')
    model.find('statistic').set('center', numbers(pivot))
    model.find('statistic').set('extent', '1.2')
    visual_lookup = {g.get('name'): g for g in visuals.getroot()}
    asset_lookup = {m.get('name'): m for m in assets.find('asset')}
    static_body = collisions.find('body')
    collider_lookup = {g.get('name'): g for g in static_body}
    report = {'source_config': cfg, 'guide_span_m': rail_span, 'carriage_span_m': moving_span,
              'travel_range_m': limits.tolist(), 'axis': cfg['axis'], 'parts': {},
              'moving_collision_names': [], 'support_collision_names': [], 'moving_visual_names': [],
              'contact_exclusion': 'Only carriage/guide-support internal contact is excluded; other body contact is retained.',
              'limitations': 'Component ownership is audited from the source mesh and user assembly description; mass, inertia and actuator settings are demonstration estimates.'}
    for name, rule in cfg['sources'].items():
        vertices, _, _, faces = meshes[name]
        lab = labels[name]
        moving_mask = np.isin(lab, rule['moving_components'])
        original_geom = visual_lookup[name]
        moving_geom = ET.fromstring(ET.tostring(original_geom))
        if moving_mask.all():
            visuals.getroot().remove(original_geom)
        else:
            moving_name = name+'_carriage'
            new_asset = ET.fromstring(ET.tostring(asset_lookup[name]))
            new_asset.set('name', moving_name)
            new_asset.set('file', f'assets/{moving_name}.msh')
            assets.find('asset').append(new_asset)
            write_subset(output/'assets'/f'{moving_name}.msh', meshes[name], moving_mask)
            write_subset(output/'assets'/f'{name}.msh', meshes[name], ~moving_mask)
            moving_geom.set('name', moving_name)
            moving_geom.set('mesh', moving_name)
        # The source meshes use world coordinates. Offset their geom frames so
        # the carriage pivot changes no visual world position at q=0.
        moving_geom.set('pos', numbers(np.fromstring(moving_geom.get('pos', '0 0 0'), sep=' ')-pivot))
        moving_geom.set('mass', '0')
        carriage.append(moving_geom)
        report['moving_visual_names'].append(moving_geom.get('name'))
        if 'collision_names' in rule:
            names = rule['collision_names']
            owners = np.full(len(names), rule['moving_components'][0])
            distances = np.zeros(len(names))
        else:
            names = [f'rmuc2027_surface_{i:05d}' for i in range(*rule['collision_index_range'])]
            centers = np.array([np.fromstring(collider_lookup[n].get('pos'), sep=' ') for n in names])
            # Search only the collider's source color mesh, avoiding ambiguity
            # with touching surfaces in other parts of the base.
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            _, distances, face_ids = trimesh.proximity.closest_point(mesh, centers)
            owners = lab[face_ids]
            if np.max(distances, initial=0) > .004:
                raise ValueError(f'{name}: collider is too far from its source surface')
        support_labels = {int(i) for i in np.unique(lab)
                          if vertices[np.unique(faces[lab == i]), 2].min() >= 1.10}
        for collider_name, owner in zip(names, owners):
            geom = collider_lookup[collider_name]
            if owner in rule['moving_components']:
                static_body.remove(geom)
                geom.set('pos', numbers(np.fromstring(geom.get('pos'), sep=' ')-pivot))
                carriage.append(geom)
                report['moving_collision_names'].append(collider_name)
            elif owner in support_labels:
                static_body.remove(geom)
                support.append(geom)
                report['support_collision_names'].append(collider_name)
        report['parts'][name] = {'moving_triangles': int(moving_mask.sum()),
                                'fixed_triangles': int((~moving_mask).sum()),
                                'maximum_collision_matching_distance_m': float(np.max(distances, initial=0))}
    if not report['moving_collision_names'] or 'visual_116_detail_0' not in report['moving_collision_names']:
        raise ValueError('Carriage must retain the green lens collider')
    for filename, tree in [('model.xml', model), ('visual_assets.xml', assets),
                           ('visual_geoms.xml', visuals), ('collision_geoms.xml', collisions)]:
        ET.indent(tree, space='  ')
        tree.write(output/filename, encoding='UTF-8', xml_declaration=True)
    report['runtime_hashes'] = runtime_hashes(output)
    if compile_model:
        start = time.monotonic()
        compiled = mujoco.MjModel.from_xml_path(str(output/'model.xml'))
        data = mujoco.MjData(compiled)
        mujoco.mj_forward(compiled, data)
        if compiled.nq != 1 or compiled.nu != 1:
            raise ValueError('Rail scene must introduce exactly one slide and one drive')
        mujoco.mj_saveModel(compiled, str(output/'model.mjb'))
        report.update(mujoco_version=mujoco.__version__, compile_seconds=time.monotonic()-start,
                      nq=compiled.nq, nu=compiled.nu)
    (output/'assembly_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/base_rail.json')
    parser.add_argument('--no-compile', action='store_true',
                        help='Export portable XML/meshes without a machine-local MJB cache')
    args = parser.parse_args()
    result = build(args.output, args.config, compile_model=not args.no_compile)
    print(json.dumps({k: result[k] for k in ['travel_range_m', 'mujoco_version', 'compile_seconds']
                      if k in result}, indent=2))


if __name__ == '__main__':
    main()
