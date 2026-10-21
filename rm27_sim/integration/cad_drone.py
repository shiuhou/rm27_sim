"""Export the supplied aircraft STEP and place one static CAD aircraft in a dock."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from .base_target import vector

from rm27_sim.paths import ROOT, resolve_asset


def build(output, scene=resolve_asset('artifacts/isolated-dock-base-20260916-v6/scene.xml'),
          config=ROOT/'configs/cad_drone.json'):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    output, scene = Path(output).resolve(), Path(scene).resolve()
    cfg = json.loads(Path(config).read_text())
    source = resolve_asset(cfg['source'])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != cfg['sha256']:
        raise ValueError('Aircraft CAD changed; review geometry and frame before updating configuration')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    parent = mujoco.MjModel.from_xml_path(str(scene))
    if parent.nq or parent.nu or any(parent.body(i).name.startswith('drone_') for i in range(parent.nbody)):
        raise ValueError('Use the empty isolated scene; aircraft already present or scene is dynamic')
    data = mujoco.MjData(parent)
    mujoco.mj_forward(parent, data)
    slot = parent.site(cfg['slot']).id
    support = parent.geom('field__collision__hangar_red_top_plate').id
    rotation = data.geom_xmat[support].reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())

    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError('STEP import failed')
    shape = reader.OneShape()
    free_faces = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)
    if free_faces.More():
        raise ValueError('Unexpected surfaces outside solids; solid-only export would omit geometry')
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds)
    lo, hi = np.array(bounds.CornerMin().Coord()), np.array(bounds.CornerMax().Coord())
    # User confirmed inverted-prop / underslung motors: CAD -Y is up.
    # Rotate 180 degrees about the previous local Y axis (no reflection),
    # retaining the camera-side direction while putting the rotors below.
    origin = np.array([(lo[0]+hi[0])/2, hi[1], (lo[2]+hi[2])/2])
    basis = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    dims = np.array([hi[0]-lo[0], hi[2]-lo[2], hi[1]-lo[1]])/1000
    if np.any(dims[:2] > .1425):
        raise ValueError(f'Aircraft does not fit slot: {dims}')
    output.mkdir(parents=True)
    assets_dir = output/'meshes'
    assets_dir.mkdir()
    BRepMesh_IncrementalMesh(shape, .08, False, .2, False).Perform()
    parts, inventory = [], []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = explorer.Current()
        index = len(parts)
        stl = assets_dir/f'part_{index:03d}.stl'
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(solid, str(stl)):
            raise RuntimeError(f'Tessellation failed: solid {index}')
        part = trimesh.load(stl, force='mesh', process=False)
        part.vertices = (part.vertices-origin) @ basis.T / 1000
        if not len(part.faces) or not np.isfinite(part.vertices).all():
            raise ValueError(f'Invalid mesh: {index}')
        b = Bnd_Box()
        BRepBndLib.AddOptimal_s(solid, b)
        inventory.append(dict(index=index, bounds_cad_mm=[b.CornerMin().Coord(), b.CornerMax().Coord()],
                              triangles=len(part.faces)))
        parts.append(part)
        explorer.Next()
    # Group by CAD index for visualization only, not a verified bill of materials.
    groups = {'electronics': range(0, 359), 'center_mount': [359],
              'rotor_units': range(360, 364), 'frame_and_camera': range(364, len(parts))}
    if len(parts) != cfg['solid_count']:
        raise ValueError('Solid ordering/count changed; re-audit groups')
    assets = ET.Element('asset')
    aircraft = ET.Element('body', name='cad_drone_1')
    for name, indices in groups.items():
        mesh = trimesh.util.concatenate([parts[i] for i in indices])
        path = assets_dir/f'{name}.obj'
        mesh.export(path)
        ET.SubElement(assets, 'mesh', name=f'cad_drone_{name}', file=str(path), maxhullvert='4')
        ET.SubElement(aircraft, 'geom', name=f'cad_drone_{name}', type='mesh', mesh=f'cad_drone_{name}',
                      rgba=cfg['display_colors'][name], contype='0', conaffinity='0', group='2')
    # A static visual body is deliberate: CAD does not specify calibrated mass,
    # stiffness, thrust or a collision decomposition for the protective cage.
    standalone = ET.Element('mujoco', model='2627drone_ver1_CAD')
    ET.SubElement(standalone, 'compiler', angle='radian')
    visual = ET.SubElement(standalone, 'visual')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.7 .7 .7')
    standalone.append(ET.fromstring(ET.tostring(assets)))
    standalone_world = ET.SubElement(standalone, 'worldbody')
    ET.SubElement(standalone_world, 'light', pos='.2 -.2 .4', dir='-.2 .2 -.4', directional='true')
    standalone_world.append(ET.fromstring(ET.tostring(aircraft)))
    ET.SubElement(standalone, 'statistic', center=f'0 0 {dims[2]/2}', extent='.18')
    ET.indent(standalone)
    ET.ElementTree(standalone).write(output/'drone.xml', encoding='unicode')

    root = ET.parse(scene).getroot()
    compiler = root.find('compiler')
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            directory = '' if compiler is None else compiler.get('meshdir' if asset.tag == 'mesh' else 'texturedir', compiler.get('assetdir', ''))
            asset.set('file', str((scene.parent/directory/asset.get('file')).resolve()))
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(key, None)
    root.find('asset').extend(list(assets))
    position = data.site_xpos[slot].copy()
    aircraft.set('pos', vector(position))
    aircraft.set('quat', vector(quat))
    root.find('worldbody').append(aircraft)
    ET.indent(root)
    ET.ElementTree(root).write(output/'scene.xml', encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    assert model.nq == model.nu == 0
    assert sum(model.body(i).name == 'cad_drone_1' for i in range(model.nbody)) == 1
    np.testing.assert_allclose(state.xpos[model.body('cad_drone_1').id], position)
    # Check actual exported triangle bounds in slot coordinates, not just CAD bounds.
    exported = np.vstack([part.vertices for part in parts])
    assert np.all(np.abs(exported[:, :2]) <= .1425/2 + 1e-6)
    assert abs(exported[:, 2].min()) < 1e-6
    for i in range(parent.ngeom):
        j = model.geom(parent.geom(i).name).id
        np.testing.assert_allclose(state.geom_xpos[j], data.geom_xpos[i], atol=1e-8)
    report = dict(passed=True, mode='headless static CAD placement; no flight or dynamic collision validation',
                  source=str(source), sha256=digest, input_scene=str(scene),
                  input_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                  units='STEP transferred to millimetres, exported to metres',
                  cad_origin_mm=origin.tolist(), cad_to_body=basis.tolist(),
                  dimensions_m=dims.tolist(), solid_count=len(parts), triangles=sum(len(p.faces) for p in parts),
                  slot=cfg['slot'], position_m=position.tolist(), quaternion=quat.tolist(),
                  side_clearance_m=((.1425-dims[:2])/2).tolist(),
                  protrusion_above_slot_m=max(0., float(dims[2]-.04)),
                  products=re.findall(r"PRODUCT\s*\(\s*'([^']*)'", source.read_text(encoding='latin1')),
                  solids=inventory, limitations=['Static visual aircraft; no mass/inertia, actuators or aircraft colliders assigned.',
                                               'Display colors are illustrative, not recovered STEP materials.',
                                               'User confirmed inverted-prop / underslung motors; CAD -Y is up. Flight-controller forward axis remains unconfirmed.',
                                               'Open-dock placement fit only; no closed-lid clearance or flight readiness claim.'])
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(output/'scene.xml')
    return report
