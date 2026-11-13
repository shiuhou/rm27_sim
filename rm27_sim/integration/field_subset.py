"""Extract reviewed 2027 structures without rebuilding or simplifying their meshes."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .field import current_field, expanded_root
from rm27_sim.paths import ROOT


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pose(element):
    if any(element.get(key) is not None for key in ('euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto')):
        raise ValueError('Subset bounds require explicit pos/quat source transforms')
    pos = np.fromstring(element.get('pos', '0 0 0'), sep=' ')
    quat = np.fromstring(element.get('quat', '1 0 0 0'), sep=' ')
    if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[pos, quat]).all() or not np.linalg.norm(quat):
        raise ValueError('Invalid source transform')
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
    return pos, rotation.reshape(3, 3)


def _vertices(mesh):
    if any(mesh.get(key) is not None for key in ('refpos', 'refquat')):
        raise ValueError('Subset bounds do not support mesh reference transforms')
    if mesh.get('vertex') is not None:
        vertices = np.fromstring(mesh.get('vertex'), sep=' ').reshape(-1, 3)
    else:
        path = Path(mesh.get('file', ''))
        if path.suffix != '.msh':
            raise ValueError(f'Spatial selection requires inline vertices or MuJoCo .msh: {path}')
        with path.open('rb') as stream:
            header = np.fromfile(stream, dtype='<i4', count=4)
            if header.size != 4 or not 0 < header[0] <= 10_000_000 or np.any(header[1:] < 0):
                raise ValueError(f'Invalid MuJoCo mesh header: {path}')
            vertices = np.fromfile(stream, dtype='<f4', count=int(header[0]) * 3)
        if vertices.size != int(header[0]) * 3:
            raise ValueError(f'Truncated MuJoCo mesh: {path}')
        vertices = vertices.reshape(-1, 3)
    scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
    if not len(vertices) or scale.shape != (3,) or not np.isfinite(vertices).all() or not np.isfinite(scale).all():
        raise ValueError('Invalid mesh vertices/scale')
    return vertices * scale


def select_subset(source, config):
    """Keep complete named structures and spatially contained collision geoms.

    No triangles are cut, decimated or made visual-only. Geometry crossing an
    ROI boundary is omitted and counted. Selected visual meshes are retained
    whole by reviewed name, independent of collision filtering.
    """
    regions = np.asarray(config['regions'], dtype=float)
    if (regions.ndim != 3 or regions.shape[1:] != (2, 3) or not len(regions) or
            not np.isfinite(regions).all() or np.any(regions[:, 0] >= regions[:, 1])):
        raise ValueError('regions must contain finite, increasing 2x3 bounds')
    keys = ('visual_geoms', 'whole_bodies', 'spatial_bodies')
    for key in keys:
        names = config[key]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names) or
                any(not isinstance(name, str) or not name for name in names)):
            raise ValueError(f'{key} must contain unique nonempty names')
    if set(config['whole_bodies']) & set(config['spatial_bodies']):
        raise ValueError('whole_bodies and spatial_bodies must be disjoint')
    if (isinstance(config['floor_z_m'], bool) or not isinstance(config['floor_z_m'], (int, float)) or
            not np.isfinite(config['floor_z_m'])):
        raise ValueError('floor_z_m must be finite')
    root = copy.deepcopy(source)
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Source has no worldbody')
    for key, tag in (('visual_geoms', 'geom'), ('whole_bodies', 'body'), ('spatial_bodies', 'body')):
        available = {item.get('name') for item in world.findall(tag)}
        missing = set(config[key]) - available
        if missing:
            raise ValueError(f'Missing requested {tag}: {sorted(missing)}')
    meshes = {item.get('name'): item for item in root.findall('asset/mesh')}
    vertices = {}
    counts = dict(source_geoms=len(list(world.iter('geom'))), retained_geoms=0,
                  spatial_retained=0, spatial_outside=0, spatial_crossing=0, omitted_heightfields=0)

    def spatial_body(body, parent_pos, parent_rotation):
        if body.find('joint') is not None or body.find('freejoint') is not None:
            raise ValueError('Dynamic bodies must be selected whole')
        pos, rotation = _pose(body)
        pos, rotation = parent_pos + parent_rotation @ pos, parent_rotation @ rotation
        for element in list(body):
            if element.tag == 'body':
                spatial_body(element, pos, rotation)
                if not list(element.iter('geom')):
                    body.remove(element)
            elif element.tag == 'geom':
                kind = element.get('type', 'sphere')
                if kind == 'hfield':
                    counts['omitted_heightfields'] += 1
                    body.remove(element)
                    continue
                if kind == 'mesh':
                    name = element.get('mesh')
                    if name not in vertices:
                        vertices[name] = _vertices(meshes[name])
                    points = vertices[name]
                elif kind == 'box':
                    size = np.fromstring(element.get('size', ''), sep=' ')
                    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                        raise ValueError('Spatial box requires three positive sizes')
                    points = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
                else:
                    raise ValueError(f'Unsupported spatial geom type: {kind}')
                geom_pos, geom_rotation = _pose(element)
                points = (points @ geom_rotation.T + geom_pos) @ rotation.T + pos
                lo, hi = points.min(axis=0), points.max(axis=0)
                contained = np.any(np.all(lo >= regions[:, 0] - 1e-7, axis=1) &
                                   np.all(hi <= regions[:, 1] + 1e-7, axis=1))
                if contained:
                    counts['spatial_retained'] += 1
                else:
                    intersects = np.any(np.all(hi >= regions[:, 0], axis=1) & np.all(lo <= regions[:, 1], axis=1))
                    counts['spatial_crossing' if intersects else 'spatial_outside'] += 1
                    body.remove(element)

    for element in list(world):
        if element.tag == 'light':
            continue
        if element.tag == 'body' and element.get('name') in config['whole_bodies']:
            continue
        if element.tag == 'body' and element.get('name') in config['spatial_bodies']:
            spatial_body(element, np.zeros(3), np.eye(3))
            if not list(element.iter('geom')):
                world.remove(element)
        elif element.tag != 'geom' or element.get('name') not in config['visual_geoms']:
            world.remove(element)
    ET.SubElement(world, 'geom', name='subset_test_floor', type='plane',
                  pos=f"0 0 {config['floor_z_m']}", size='0 0 .1', rgba='.22 .24 .26 1',
                  contype='1', conaffinity='1', friction='.8 .01 .001')
    # Preserve only relationships whose endpoints remain in the selected model.
    names = {item.get('name') for item in world.iter() if item.get('name')}
    for section in root.findall('actuator') + root.findall('contact'):
        for element in list(section):
            references = [value for key, value in element.attrib.items()
                          if key in ('joint', 'site', 'body', 'body1', 'body2', 'geom1', 'geom2', 'tendon')]
            if any(name not in names for name in references):
                section.remove(element)
    assets = [item for section in root.findall('asset') for item in section]
    asset_ids = {id(item) for item in assets}
    needed = {tag: set() for tag in ('mesh', 'material', 'texture', 'hfield')}
    for item in root.iter():
        if id(item) not in asset_ids:
            for tag in needed:
                if item.get(tag):
                    needed[tag].add(item.get(tag))
    for item in assets:
        if item.tag == 'material' and item.get('name') in needed['material'] and item.get('texture'):
            needed['texture'].add(item.get('texture'))
    for section in root.findall('asset'):
        for item in list(section):
            if item.tag in needed and item.get('name') not in needed[item.tag]:
                section.remove(item)
    counts.update(retained_geoms=len(list(world.iter('geom'))), source_assets=len(assets),
                  retained_assets=sum(len(section) for section in root.findall('asset')))
    center = (regions[:, 0].min(axis=0) + regions[:, 1].max(axis=0)) / 2
    statistic = root.find('statistic')
    if statistic is not None:
        statistic.set('center', ' '.join(map(str, center)))
        statistic.set('extent', str(float(np.max(regions[:, 1].max(0) - regions[:, 0].min(0)))))
    root.set('model', 'rm27_2027_base_dock_subset')
    return root, counts


def add_aircraft(root, settings):
    """Reuse the packaged aircraft; describe parked and airborne initial states.

    Placement is scene initialization, not a hover controller or takeoff result.
    Source H paint does not establish a team assignment for either native tray.
    """
    from . import drone
    from rm27_sim.paths import resolve_asset
    required = {'config', 'dock_body', 'team', 'start_height_above_slot_m'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f'aircraft settings must contain {sorted(required)}')
    height = settings['start_height_above_slot_m']
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not np.isfinite(height) or not .3 <= height <= 5:
        raise ValueError('aircraft start height above slot must be in [.3, 5] metres')
    if settings['team'] not in ('red', 'blue'):
        raise ValueError('aircraft team must be red or blue')
    dock = root.find(f"worldbody/body[@name='{settings['dock_body']}']")
    if dock is None:
        raise ValueError('aircraft dock_body must be a retained native dock')
    config_path = ROOT / settings['config']
    config = json.loads(config_path.read_text())
    if config.get('model') != 'cad_v2':
        raise ValueError('Use the current Ver2 aircraft configuration')
    package = resolve_asset(config['runtime_package'])
    template = package / 'dynamic/model.xml'
    sources = [config_path, package / 'model_info.json', template]
    sources += [(template.parent / item.get('file')).resolve() for item in ET.parse(template).findall('asset/*') if item.get('file')]
    hashes = {str(path): _hash(path) for path in sources}
    position, rotation = _pose(dock)
    quaternion = np.fromstring(dock.get('quat', '1 0 0 0'), sep=' ')
    parked, airborne, vehicles, slots = [], [], [], {}
    for index in range(1, 5):
        slot = dock.find(f"site[@name='{settings['dock_body']}_slot_{index}']")
        if slot is None:
            raise ValueError(f'Native dock is missing slot {index}')
        local, _ = _pose(slot)
        slots[f'slot_{index}'] = (position + rotation @ local).tolist()
        resting = local + [0, 0, config['com_height_above_bottom_m_assumed'] + .001]
        start = resting.copy()
        if index == 1:
            start[2] = local[2] + height
        spawn = position + rotation @ start
        vehicles.append(drone.add(root, f'drone_{index}', spawn, quaternion, config))
        parked.extend([*(position + rotation @ resting), *quaternion])
        airborne.extend([*spawn, *quaternion])
        # Preserve native names and also expose the shared MicroPlant slot contract.
        alias = copy.deepcopy(slot)
        alias.set('name', f'slot_{index}')
        dock.append(alias)
    joints = list(root.find('worldbody').iter('joint'))
    if len(joints) != 1 or joints[0].get('name') != 'blue_base_slide':
        raise ValueError('Expected the original independent blue base rail')
    keyframe = ET.SubElement(root, 'keyframe')
    for name, values in (('parked', parked), ('mission_start', airborne)):
        key = ET.SubElement(keyframe, 'key', name=name, qvel=' '.join(['0'] * 25),
                            ctrl=' '.join(['0'] * 17), act=' '.join(['0'] * 16))
        if name == 'parked':
            key.set('qpos', ' '.join(map(str, [0., *values])))
        # Omitted qpos inherits compiled qpos0, including the aircraft builder's
        # serialization precision and MuJoCo's quaternion normalization.
    return dict(vehicles=vehicles, team=settings['team'],
                team_assignment='simulation configuration; native dock team unverified (both H borders are blue)',
                dock_body=settings['dock_body'], slots_world_m=slots, source_sha256=hashes,
                default_state='mission_start', active_aircraft='drone_1',
                parked_reference_keyframe='parked', start_height_above_slot_m=height,
                mission_start_world_m=airborne[:3], scope='initial scene placement only; no takeoff/hover qualification')


def extract(config_path, output, *, preview=False):
    """Create an exclusive, self-contained model with exact copied dependencies."""
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    config = json.loads(config_path.read_text())
    path, field_config, manifest = current_field(config=ROOT / config['field_config'])
    if path.parent == output or path.parent in output.parents:
        raise ValueError('output must be outside the original assets')
    started = time.monotonic()
    root, counts = select_subset(expanded_root(path), config)
    aircraft = add_aircraft(root, config['aircraft']) if 'aircraft' in config else None
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(passed=False, status='IN_PROGRESS', scope='2027 base and hangar scene extraction; no flight control',
                   config=config, configuration_sha256=_hash(config_path), source_config=field_config,
                   source_runtime_sha256=manifest['runtime_hashes'], original_inputs_unchanged=False,
                   mujoco_version=mujoco.__version__, code_sha256=_hash(__file__), counts=counts,
                   copied_assets={}, rendering='NOT_RUN', flight_control='NOT_RUN',
                   aircraft=aircraft,
                   limitations=['World poses, selected original collision geoms and rail dynamics are preserved.',
                                'ROIs omit whole crossing geoms and remote field terrain; this is not the full arena.',
                                'A flat test floor is added at the declared height; removed terrain is not reconstructed.',
                                'No decimation, remeshing or historical 2026 base assets are used.'])
    try:
        assets_dir = output / 'assets'
        assets_dir.mkdir()
        for element in root.iter():
            if element.get('file'):
                source = Path(element.get('file'))
                digest = _hash(source)
                relative = f'assets/{digest}{source.suffix}'
                destination = output / relative
                if not destination.exists():
                    shutil.copyfile(source, destination)
                if _hash(destination) != digest:
                    raise OSError(f'Copied asset hash mismatch: {source}')
                summary['copied_assets'][relative] = dict(source=str(source), sha256=digest, bytes=source.stat().st_size)
                element.set('file', relative)
        scene = output / 'scene.xml'
        ET.ElementTree(root).write(scene, encoding='unicode')
        summary.update(scene='scene.xml', scene_sha256=_hash(scene))
        compile_started = time.monotonic()
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        summary.update(compile_wall_s=time.monotonic() - compile_started, geoms=model.ngeom,
                       meshes=model.nmesh, nq=model.nq, nv=model.nv, nu=model.nu,
                       initial_contacts=data.ncon,
                       sites_world_m={model.site(i).name: data.site_xpos[i].tolist() for i in range(model.nsite)})
        mujoco.mj_saveModel(model, str(output / 'scene.mjb'))
        summary.update(compiled_model='scene.mjb', compiled_model_sha256=_hash(output / 'scene.mjb'),
                       compiled_bytes=(output / 'scene.mjb').stat().st_size)
        if aircraft is None:
            for _ in range(500):
                mujoco.mj_step(model, data)
        else:
            mujoco.mj_resetDataKeyframe(model, data, model.key('mission_start').id)
            mujoco.mj_forward(model, data)
            if data.ncon:
                raise ValueError('Initial airborne/parked aircraft intersect the extracted scene')
            summary['aircraft']['default_qpos_matches_mission_start'] = bool(np.array_equal(data.qpos, model.qpos0))
            if not summary['aircraft']['default_qpos_matches_mission_start']:
                raise ValueError('Default state must equal the declared airborne mission start')
            summary['aircraft']['initial_motor_controls_zero'] = bool(not np.any(data.ctrl))
            summary['aircraft']['dynamics_validation'] = 'NOT_RUN; unsupported free-fall is not a hover test'
        if not np.isfinite(np.r_[data.qpos, data.qvel, data.geom_xpos.ravel()]).all():
            raise RuntimeError('Nonfinite extracted scene state')
        if aircraft is None:
            summary['uncontrolled_one_second_max_displacement_m'] = float(np.max(np.abs(data.qpos - model.qpos0), initial=0))
        if preview:
            from PIL import Image
            with mujoco.Renderer(model, height=480, width=640,
                                  max_geom=max(10000, model.ngeom + model.nsite + 1000)) as renderer:
                views = [('overview', np.asarray(model.stat.center), 21., 110., -22.)]
                for index, bounds in enumerate(config['regions']):
                    bounds = np.asarray(bounds)
                    views.append((('base', 'dock')[index], bounds.mean(0),
                                  float(np.max(bounds[1] - bounds[0])) * 1.6,
                                  (180., 225.)[index], -18.))
                if aircraft is not None:
                    dock_body = model.body(aircraft['dock_body']).id
                    outward_azimuth = float(np.degrees(np.arctan2(-data.xpos[dock_body, 1], -data.xpos[dock_body, 0])))
                    views += [('parking', data.xpos[dock_body] + [0, 0, .18], .7, outward_azimuth, -48.),
                              ('airborne', np.asarray(aircraft['mission_start_world_m']), .65, 145., -20.)]
                for name, center, distance, azimuth, elevation in views:
                    if aircraft is not None:
                        state = 'parked' if name == 'parking' else 'mission_start'
                        mujoco.mj_resetDataKeyframe(model, data, model.key(state).id)
                        mujoco.mj_forward(model, data)
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = center
                    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
                    renderer.update_scene(data, camera=camera)
                    Image.fromarray(renderer.render()).save(output / f'{name}.png')
            summary['rendering'] = 'PASS'
        checked_path, checked_config, checked_manifest = current_field(config=ROOT / config['field_config'])
        if (checked_path != path or checked_config != field_config or checked_manifest != manifest or
                _hash(config_path) != summary['configuration_sha256'] or _hash(__file__) != summary['code_sha256'] or
                (aircraft is not None and any(_hash(name) != digest for name, digest in aircraft['source_sha256'].items()))):
            raise ValueError('Source/configuration/code changed during extraction')
        summary.update(passed=True, status='PASS', original_inputs_unchanged=True)
    except BaseException as exc:
        summary.update(status='FAIL', failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['elapsed_wall_s'] = time.monotonic() - started
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary
