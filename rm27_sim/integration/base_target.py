"""Compose a provisional base marker and contact patch without replacing the field."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rm27_sim.paths import ROOT, resolve_asset


def vector(value):
    return ' '.join(str(float(v)) for v in value)


def add(root, cfg):
    if any(e.get('name', '').startswith('rm27_base') for e in root.iter()):
        raise ValueError('Base target already present')
    if cfg.get('friendly_team') != 'red' or cfg.get('target_team') != 'blue':
        raise ValueError('This mission requires the red aircraft to target the blue base')
    p = np.asarray(cfg['armor_surface_center_m'], float)
    n = np.asarray(cfg['armor_normal'], float)
    if p.shape != (3,) or n.shape != (3,) or not np.isfinite([p, n]).all() or np.linalg.norm(n) < 1e-8:
        raise ValueError('Invalid target position/normal')
    n /= np.linalg.norm(n)
    width, length, thickness = cfg['armor_dimensions_m']
    if min(width, length, thickness, cfg['marker_radius_m']) <= 0:
        raise ValueError('Target dimensions must be positive')
    # Local X is horizontal across the plate; local Z points outward.
    x = np.cross([0, 0, 1], n)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('Target normal must have a horizontal component')
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, n]).ravel())
    body = ET.SubElement(root.find('worldbody'), 'body', name='rm27_base_frame')
    ET.SubElement(body, 'geom', name='rm27_base_armor', type='box',
                  pos=vector(p-n*thickness/2), quat=vector(quat),
                  size=vector([width/2, length/2, thickness/2]),
                  rgba='0.02 0.45 0.95 1', group='0', contype='1', conaffinity='1',
                  friction='.8 .01 .001', solref='.01 1', solimp='.95 .99 .001')
    ET.SubElement(body, 'site', name='rm27_base_armor_center', pos=vector(p),
                  quat=vector(quat), size='.002', rgba='0 0 0 0')
    ET.SubElement(body, 'site', name='rm27_base_marker_center',
                  pos=vector(cfg['marker_center_m']), size='.002', rgba='0 0 0 0')
    ET.SubElement(body, 'geom', name='rm27_base_marker', type='cylinder',
                  pos=vector(cfg['marker_center_m']), zaxis=vector(cfg['marker_normal']),
                  size=vector([cfg['marker_radius_m'], .002]),
                  rgba='0.02 1 0.03 1', contype='0', conaffinity='0', group='0')
    return p, n


def build(scene, output, config=ROOT/'configs/base_target.json'):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    cfg = json.loads(Path(config).read_text())
    field = resolve_asset(cfg['field_path'])
    if hashlib.sha256(field.read_bytes()).hexdigest() != cfg['field_sha256']:
        raise ValueError('Field changed; review the target placement')
    root = ET.parse(scene).getroot()
    # Resolve file references against the input compiler's asset directories.
    compiler = root.find('compiler')
    for e in root.findall('asset/*'):
        if e.get('file'):
            directory = '' if compiler is None else compiler.get('meshdir' if e.tag == 'mesh' else 'texturedir', compiler.get('assetdir', ''))
            e.set('file', str((scene.parent/directory/e.get('file')).resolve()))
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(key, None)
    original = mujoco.MjModel.from_xml_path(str(scene))
    reference = mujoco.MjModel.from_xml_path(str(field))
    reference_data = mujoco.MjData(reference)
    original_data = mujoco.MjData(original)
    mujoco.mj_forward(reference, reference_data)
    mujoco.mj_forward(original, original_data)
    for i in range(reference.ngeom):
        name = reference.geom(i).name
        j = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_GEOM, name)
        if j < 0:
            raise ValueError(f'Input scene missing field geom: {name}')
        for a, b in ((reference_data.geom_xpos[i], original_data.geom_xpos[j]),
                     (reference_data.geom_xmat[i], original_data.geom_xmat[j]),
                     (reference.geom_size[i], original.geom_size[j])):
            if not np.allclose(a, b, atol=1e-8, rtol=0):
                raise ValueError(f'Input field geometry changed: {name}')
    p, n = add(root, cfg)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    if model.nu != original.nu or model.nq != original.nq:
        raise ValueError('Target changed vehicle degrees of freedom')
    output.mkdir(parents=True, exist_ok=False)
    (output/'scene.xml').write_text(xml)
    report = dict(configuration=cfg, input_scene=str(scene), input_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                  compiled=True, added_geoms=model.ngeom-original.ngeom,
                  friendly_team=cfg['friendly_team'], target_team=cfg['target_team'],
                  armor_surface_center_m=p.tolist(), armor_normal=n.tolist(),
                  status='provisional static target; no flight/contact mission certified',
                  limitations=['Marker world placement is provisional; diameter and optical specifications come from the RMUC 2026 rulebook.',
                               'Armor patch lies on a candidate mesh surface; target identity is provisional.',
                               'Only armor patch collides; surrounding base is still visual-only.',
                               'Green geometry does not model 520 nm emission, approximately 10 cd intensity, illumination state, bloom, exposure or a camera.'])
    (output/'target.json').write_text(json.dumps(report, indent=2))
    print(output/'scene.xml')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/base_target.json')
    args=parser.parse_args(); build(args.scene,args.output,args.config)

if __name__ == '__main__':
    main()
