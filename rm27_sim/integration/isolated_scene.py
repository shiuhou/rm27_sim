"""Extract the current blue base and dock into a static, empty test world."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from .base_target import add, vector

from rm27_sim.paths import ROOT, resolve_asset


def contained_components(vertices, faces, bounds):
    """Select complete connected surfaces; never cut triangles at the ROI."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    lo, hi = np.asarray(bounds)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    parts = mesh.submesh(groups, repair=False)
    return [part for part in parts
            if np.all(part.bounds[0] >= lo) and np.all(part.bounds[1] <= hi)]


def build(scene, output, config=ROOT/'configs/isolated_scene.json'):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    cfg = json.loads(Path(config).read_text())
    target = json.loads((ROOT/'configs/base_target.json').read_text())
    source_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
    if source_hash != cfg['source_scene_sha256']:
        raise ValueError('Source scene changed; review the extraction region and source hash first')
    source_root = ET.parse(scene).getroot()
    compiler = source_root.find('compiler')
    asset_hashes = {}
    for asset in source_root.findall('asset/*'):
        if asset.get('file'):
            directory = '' if compiler is None else compiler.get('meshdir' if asset.tag == 'mesh' else 'texturedir', compiler.get('assetdir', ''))
            path = (scene.parent/directory/asset.get('file')).resolve()
            asset_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for name in ('rm27_dock', 'center_hangar_red'):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f'Missing body: {name}')
    output.mkdir(parents=True)
    root = ET.Element('mujoco', model='rm27_isolated_dock_base')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.002', integrator='RK4', gravity='0 0 -9.81')
    ET.SubElement(root, 'statistic', center='-5.5 0.6 0.8', extent='15')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1200', offheight='800')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.7 .7 .7')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='0.18 0.28 0.40', rgb2='0.02 0.03 0.05', width='512', height='3072')
    ET.SubElement(assets, 'texture', name='grid', type='2d', builtin='checker',
                  rgb1='0.25 0.25 0.25', rgb2='0.35 0.35 0.35', width='512', height='512')
    ET.SubElement(assets, 'material', name='ground', texture='grid',
                  texrepeat='1 1', texuniform='true', reflectance='0.08')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='-5 0 8', directional='true', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='test_floor', type='plane', size='20 10 .1', material='ground')
    retained, selected = [], []

    def vertices(i):
        j = m.geom_dataid[i]
        v = m.mesh_vert[m.mesh_vertadr[j]:m.mesh_vertadr[j]+m.mesh_vertnum[j]]
        f = m.mesh_face[m.mesh_faceadr[j]:m.mesh_faceadr[j]+m.mesh_facenum[j]]
        return v @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i], f

    def mesh_geom(name, part, rgba):
        path = output/f'{name}.obj'
        part.export(path)
        ET.SubElement(assets, 'mesh', name=name, file=str(path), maxhullvert='4')
        ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name,
                      rgba=vector(rgba), contype='0', conaffinity='0', group='2')

    for i in range(m.ngeom):
        name = m.geom(i).name
        body = m.body(m.geom_bodyid[i]).name
        if body in ('rm27_dock', 'center_hangar_red'):
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                v, f = vertices(i)
                mesh_geom(name, trimesh.Trimesh(v, f, process=False), m.geom_rgba[i])
            else:
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, d.geom_xmat[i])
                ET.SubElement(world, 'geom', name=name, type='box',
                              pos=vector(d.geom_xpos[i]), quat=vector(quat), size=vector(m.geom_size[i]),
                              rgba=vector(m.geom_rgba[i]), group=str(m.geom_group[i]),
                              contype=str(m.geom_contype[i]), conaffinity=str(m.geom_conaffinity[i]),
                              friction=vector(m.geom_friction[i]), solref=vector(m.geom_solref[i]),
                              solimp=vector(m.geom_solimp[i]))
            retained.append(name)
        elif name.startswith('field__visual__') and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            v, f = vertices(i)
            parts = contained_components(v, f, cfg['base_bounds_m'])
            if parts:
                part = trimesh.util.concatenate(parts)
                extracted = f'isolated_{name}'
                material = m.geom_matid[i]
                color = m.mat_rgba[material] if material >= 0 else m.geom_rgba[i]
                mesh_geom(extracted, part, color)
                selected.append(dict(source_geom=name, name=extracted, triangles=len(part.faces), bounds_m=part.bounds.tolist()))
    if not selected:
        raise ValueError('No base components selected')
    # The original column starts on a 0.4 m platform. Keep its height/pose and
    # provide a clearly separate footing in the empty world.
    i = m.geom('field__collision__hangar_red_column').id
    bottom = d.geom_xpos[i, 2]-m.geom_size[i, 2]
    if bottom > 0:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.geom_xmat[i])
        ET.SubElement(world, 'geom', name='test_column_footing', type='box',
                      pos=vector([*d.geom_xpos[i, :2], bottom/2]), quat=vector(quat),
                      size=vector([*m.geom_size[i, :2], bottom/2]), rgba='.48 .48 .42 1')
    for name in ('dock_frame', 'slot_1', 'slot_2', 'slot_3', 'slot_4'):
        ET.SubElement(world, 'site', name=name, pos=vector(d.site_xpos[m.site(name).id]), size='.001', rgba='0 0 0 0')
    add(root, target)
    ET.indent(root)
    path = output/'scene.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    result = mujoco.MjModel.from_xml_path(str(path))
    state = mujoco.MjData(result)
    for _ in range(100):
        mujoco.mj_step(result, state)
    assert result.nq == result.nu == 0
    assert np.isfinite(state.geom_xpos).all()
    for name in retained:
        if m.geom(name).type != mujoco.mjtGeom.mjGEOM_MESH:
            np.testing.assert_allclose(state.geom_xpos[result.geom(name).id], d.geom_xpos[m.geom(name).id], atol=1e-8)
    report = dict(passed=True, mode='headless static model validation', source_scene=str(scene),
                  source_sha256=source_hash, source_asset_sha256=asset_hashes,
                  configuration=cfg, target_configuration=target,
                  retained_geoms=retained, base_components=selected, geoms=result.ngeom,
                  colliders=int(np.count_nonzero(result.geom_contype | result.geom_conaffinity)),
                  column_footing_height_m=float(bottom),
                  limitations=['Current field blue base extracted by whole connected components in a reviewed bounding region; not a new base2.STEP conversion.',
                               'Base tower is visual-only; only the existing provisional armor patch collides.',
                               'No aircraft or flight controller. Full-field route audits do not apply.'])
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(path)
    return report
