"""Verified current development field; historical assets keep their own configs."""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

from rm27_sim.paths import ROOT, resolve_asset

CONFIG = ROOT / 'configs/field.json'


def current_field(asset_root=None, config=CONFIG):
    cfg = json.loads(Path(config).read_text())
    path = (Path(asset_root) / Path(cfg['path']).relative_to('components/rm_mjcf')
            if asset_root is not None else resolve_asset(cfg['path']))
    path = path.resolve()
    manifest_path = path.parent / 'assembly_manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != cfg['manifest_sha256']:
        raise ValueError('Current field manifest changed; review configs/field.json')
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['runtime_hashes'].items():
        if hashlib.sha256((path.parent/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Current field asset changed: {name}')
    return path, cfg, manifest


def expanded_root(path):
    """Expand MJCF includes and resolve asset paths before writing a scene elsewhere."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    assetdir = compiler.get('assetdir', '') if compiler is not None else ''
    meshdir = compiler.get('meshdir', assetdir) if compiler is not None else assetdir
    texturedir = compiler.get('texturedir', assetdir) if compiler is not None else assetdir

    def expand(parent, directory):
        for child in list(parent):
            if child.tag == 'include':
                source = (directory/child.get('file')).resolve()
                included = ET.parse(source).getroot()
                expand(included, source.parent)
                index = list(parent).index(child)
                parent.remove(child)
                for offset, item in enumerate(included):
                    parent.insert(index+offset, item)
            else:
                if child.get('file'):
                    subdir = meshdir if child.tag == 'mesh' else texturedir if child.tag == 'texture' else assetdir
                    child.set('file', str((directory/subdir/child.get('file')).resolve()))
                expand(child, directory)
    expand(root, path.parent)
    if compiler is not None:
        for key in ('assetdir', 'meshdir', 'texturedir'):
            compiler.attrib.pop(key, None)
    return root


def compose_development(cfg, drone_cfg, output, *, camera=None, save_compiled=False):
    """Reuse the existing dock, preserving field contacts and the rail actuator."""
    import mujoco
    import numpy as np
    from . import drone
    path, source, manifest = current_field(config=ROOT/cfg['field']['config'])
    root = expanded_root(path)
    name = cfg['dock']['existing_body']
    body = root.find(f"worldbody/body[@name='{name}']")
    if body is None:
        raise ValueError(f'Current field is missing dock {name}')
    position = np.fromstring(body.get('pos'), sep=' ')
    quat = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
    rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, quat); rotation = rotation.reshape(3, 3)
    body.set('name', 'rm27_dock')
    for original in list(body.findall('site')):
        alias = copy.deepcopy(original)
        alias.set('name', original.get('name').removeprefix(name+'_'))
        body.append(alias)
    vehicles = []
    for i in range(1, 5):
        slot = body.find(f"site[@name='slot_{i}']")
        local = np.fromstring(slot.get('pos'), sep=' ')
        local[2] += drone_cfg['com_height_above_bottom_m_assumed'] + .001
        vehicles.append(drone.add(root, f'drone_{i}', position+rotation@local, quat, drone_cfg))
    if camera is not None:
        attributes = dict(camera)
        camera_body = attributes.pop('body')
        vehicle_body = root.find(f"worldbody/body[@name='{camera_body}']")
        if camera_body not in {item['name'] for item in vehicles} or vehicle_body is None:
            raise ValueError(f'Camera requires a composed vehicle body: {camera_body}')
        ET.SubElement(vehicle_body, 'camera', **attributes)
        # The PX4 range adapter excludes group 5 to avoid measuring aircraft.
        # Group selection changes ray/render visibility, not collision masks.
        for vehicle in vehicles:
            for geom in root.find(f"worldbody/body[@name='{vehicle['name']}']").iter('geom'):
                geom.set('group', '5')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = output/'scene.xml'
    ET.ElementTree(root).write(scene, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    if data.ncon:
        raise ValueError('Initial contacts in development scene; inspect placement')
    result = dict(field_commit=source['source_commit'], field_path=str(path),
                  field_manifest_sha256=source['manifest_sha256'],
                  field_model_sha256=manifest['runtime_hashes']['model.xml'],
                  dock_world_position_m=position.tolist(), dock_world_quaternion=quat.tolist(),
                  placement=cfg['dock']['placement_note'], scene_geoms=model.ngeom,
                  slots_world_m={f'slot_{i}':data.site_xpos[model.site(f'slot_{i}').id].tolist() for i in range(1,5)},
                  vehicles=vehicles, target_default='hold_center',
                  mission_status='Composition only; flight/route/SITL results require new verification')
    if save_compiled:
        compiled = output/'scene.mjb'
        mujoco.mj_saveModel(model, str(compiled))
        digest = hashlib.sha256()
        with compiled.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        result.update(compiled_model=str(compiled), compiled_model_sha256=digest.hexdigest(),
                      mujoco_version=mujoco.__version__)
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
