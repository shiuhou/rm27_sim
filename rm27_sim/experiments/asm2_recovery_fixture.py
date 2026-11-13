"""Native ASM2 on a neutral inclined slab for independent recovery tests."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rm27_sim.integration.field import expanded_root
from rm27_sim.paths import ROOT
from rm27_sim.simulation.asm2 import digest, write_json

AIRCRAFT = ROOT / 'data/shared_aircraft/aircraft.json'
AIRCRAFT_XML_SHA256 = '2356bc3f8372fe14002521a82d4a9533867c688f8ed3d13cf172c74e17905ae4'
METADATA_SHA256 = 'f4a5c52e8c06de893d3dc99b5544ba4408f03eed812bee52b595ee52db860b08'
PANEL_NAME = 'fixture_panel'
PANEL_HALF_SIZE_M = (.36, .36, .02)
PANEL_INCLINE_DEG = 18.


def _numbers(values):
    return ' '.join(str(float(value)) for value in values)


def load_fixture(output: Path):
    """Compile the registered aircraft with one fixed, neutral test surface."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene_xml, scene_mjb, provenance_path = (output / name for name in
                                            ('scene.xml', 'scene.mjb', 'scene_provenance.json'))
    if any(path.exists() for path in (scene_xml, scene_mjb, provenance_path)):
        raise FileExistsError(f'Fixture evidence already exists in {output}')
    registry = json.loads(AIRCRAFT.read_text())
    entry = registry['entries']['default']
    xml = AIRCRAFT.parent / entry['path']
    metadata_path = AIRCRAFT.parent / registry['plant_metadata']
    release_manifest = (AIRCRAFT.parent / registry['provenance']).resolve()
    if (registry['aircraft'], registry['release'], registry['mujoco_proxy']) != ('ASM2', '1.0.0', '0.1.0'):
        raise ValueError('The fixture requires the frozen ASM2 aircraft release')
    if digest(xml) != AIRCRAFT_XML_SHA256 or entry['sha256'] != AIRCRAFT_XML_SHA256 or digest(metadata_path) != METADATA_SHA256:
        raise ValueError('ASM2 model or plant metadata hash mismatch')
    if mujoco.__version__ != registry['mujoco_version']:
        raise ValueError('Fixture MuJoCo runtime differs from the registered aircraft version')
    metadata = json.loads(metadata_path.read_text())

    root = expanded_root(xml)
    release = json.loads(release_manifest.read_text())
    if release['payload_sha256'] != registry['archive_payload_sha256']:
        raise ValueError('Aircraft release manifest does not match the registry')
    assets = {Path(asset.attrib['file']) for asset in root.findall('asset/*') if 'file' in asset.attrib}
    verified = {}
    for asset in sorted(assets):
        key = asset.relative_to(release_manifest.parent).as_posix()
        actual = digest(asset)
        if release['files'].get(key) != actual:
            raise ValueError(f'Frozen ASM2 mesh hash mismatch: {asset}')
        verified[str(asset)] = actual
    for path in (AIRCRAFT, xml, metadata_path, release_manifest):
        verified[str(path)] = digest(path)

    root.set('model', 'asm2_neutral_recovery_fixture')
    world = root.find('worldbody')
    body = world.find('body')
    body.set('name', 'uav')
    body.find('freejoint').set('name', 'asm2_free')
    # Keep the default body position clear of the slab before a case is set.
    body.set('pos', '0 0 .30')
    frame = Rotation.from_euler('y', -PANEL_INCLINE_DEG, degrees=True)
    slope, across, normal = frame.as_matrix().T
    center = np.zeros(3)
    ET.SubElement(
        world, 'geom', name=PANEL_NAME, type='box', size=_numbers(PANEL_HALF_SIZE_M),
        pos=_numbers(center - PANEL_HALF_SIZE_M[2] * normal),
        quat=_numbers(frame.as_quat(scalar_first=True)), rgba='.58 .61 .64 1',
        priority='1', solref='.004 1', friction='.8 .005 .0001', condim='3',
        margin='0', gap='0',
    )
    ET.ElementTree(root).write(scene_xml, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    aircraft_id = model.body('uav').id
    aircraft_collision = (model.geom_bodyid == aircraft_id) & (model.geom_contype | model.geom_conaffinity).astype(bool)
    if (model.body_mass[aircraft_id] != metadata['mass_kg'] or model.nu != entry['actuators']
            or model.opt.timestep != metadata['contact_dt_s']
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or np.count_nonzero(aircraft_collision) != entry['collision_geoms']):
        raise ValueError('Compiled fixture did not preserve the aircraft dynamics and bindings')
    mujoco.mj_saveModel(model, str(scene_mjb))

    # ContactScorer reads these legacy field names as generic surface coordinates.
    # The values are measured from this slab, with no field geometry or field inputs.
    mapping = dict(
        armor_surface_center_m=center.tolist(), armor_normal=normal.tolist(),
        slope_axis=slope.tolist(), across_axis=across.tolist(),
        contact_region_halfsize_slope_across_m=list(PANEL_HALF_SIZE_M[:2]),
        registered_contact_geoms=[PANEL_NAME],
    )
    provenance = dict(
        compiled_sha256=digest(scene_mjb), scene_xml_sha256=digest(scene_xml),
        source_sha256=verified,
        mujoco_version=mujoco.__version__, physics_dt_s=float(model.opt.timestep),
        native_aircraft_collision_geoms=int(np.count_nonzero(aircraft_collision)),
        surface=dict(name=PANEL_NAME, incline_deg=PANEL_INCLINE_DEG, half_size_m=PANEL_HALF_SIZE_M,
                     time_constant_s=.004, damping_ratio=1., friction=[.8, .005, .0001], condim=3),
        mapping=mapping,
    )
    write_json(provenance_path, provenance)
    return model, metadata, mapping, provenance


def _minimum_projection(model, body, plane_normal_body):
    colliders = np.flatnonzero(
        (model.geom_bodyid == body) & (model.geom_contype | model.geom_conaffinity).astype(bool))
    if not len(colliders) or np.any(model.geom_type[colliders] != mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError('Expected native ASM2 aircraft collision meshes')
    orientations = Rotation.from_quat(model.geom_quat[colliders], scalar_first=True)
    normals = orientations.inv().apply(plane_normal_body)
    offsets = model.geom_pos[colliders] @ plane_normal_body
    support = []
    for index, mesh in enumerate(model.geom_dataid[colliders]):
        start, length = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        support.append(float(np.min(model.mesh_vert[start:start + length] @ normals[index]) + offsets[index]))
    return min(support)


def initialize(plant, scorer, case):
    """Set one separated release at time zero before any native physics step."""
    gap = float(case['gap_m'])
    inward = float(case['inward_speed_m_s'])
    tangential = float(case['tangential_speed_m_s'])
    pitch = float(case['tilt_offset_deg'])
    roll = float(case['roll_offset_deg'])
    lateral = float(case.get('lateral_offset_m', 0.))
    rates = np.asarray(case['omega_body_rad_s'], dtype=float)
    hover = float(case['hover_fraction']) * plant.nominal_mass * 9.81 / 4.
    if (rates.shape != (3,) or not np.isfinite([gap, inward, tangential, pitch, roll, lateral, hover, *rates]).all()
            or gap <= 0. or inward < 0. or not 0. <= hover <= plant.max_motor):
        raise ValueError('Fixture release has invalid gap, velocity, attitude, rates, or rotor activation')
    if plant.data.time != 0.:
        raise ValueError('Fixture initialization requires a new plant before the first solver step')

    frame = Rotation.from_matrix(np.column_stack((*scorer.axes, scorer.normal)))
    orientation = frame * Rotation.from_euler('YX', [pitch, roll], degrees=True)
    minimum = _minimum_projection(plant.model, plant.body, orientation.inv().apply(scorer.normal))
    position = scorer.center + (gap - minimum) * scorer.normal + lateral * scorer.axes[1]
    velocity = -inward * scorer.normal + tangential * scorer.axes[0]
    quaternion = orientation.as_quat(scalar_first=True)
    plant.data.qpos[plant.qadr:plant.qadr + 7] = np.r_[position, quaternion]
    plant.data.qvel[plant.dadr:plant.dadr + 6] = np.r_[velocity, rates]
    plant.data.act[plant.activation] = hover
    plant.data.ctrl[plant.motors] = hover
    mujoco.mj_forward(plant.model, plant.data)
    if plant.data.ncon:
        raise ValueError('Fixture release starts in native collision contact')
    return dict(
        initial_clearance_m=gap, position_m=position.tolist(), velocity_m_s=velocity.tolist(),
        omega_body_rad_s=rates.tolist(), body_to_world_quat_wxyz=quaternion.tolist(),
        held_motor_command_N=plant.data.ctrl[plant.motors].tolist(),
    )
