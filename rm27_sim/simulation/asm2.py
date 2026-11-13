"""Offline ASM2 controlled-contact and recovery comparison on native field geometry."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rm27_sim.integration.field import current_field, expanded_root
from rm27_sim.integration.field_subset import select_subset
from rm27_sim.paths import ROOT
from rm27_sim.experiments.contact_metrics import point_approach_speed

DEFAULT_CONFIG = ROOT / 'configs/asm2_contact_recovery.json'
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compose_scene(config, output, full_field=False):
    """Combine source meshes, native contacts, and the original rotor actuators."""
    registry_path = ROOT / config['aircraft']
    registry = json.loads(registry_path.read_text())
    entry = registry['entries']['default']
    xml = registry_path.parent / entry['path']
    metadata_path = registry_path.parent / registry['plant_metadata']
    if digest(xml) != entry['sha256']:
        raise ValueError('ASM2 model hash mismatch')
    metadata = json.loads(metadata_path.read_text())
    field, _, manifest = current_field()
    target_path = ROOT / config['target_mapping']
    target = json.loads(target_path.read_text())
    if digest(field) != target['field_sha256']:
        raise ValueError('Field differs from the registered target')
    root = expanded_root(field)
    counts = None
    if not full_field:
        subset = json.loads((ROOT / config['field_subset']).read_text())
        index = config['subset_region_index']
        subset['regions'] = [subset['regions'][index]]
        subset['whole_bodies'] = [name for name in subset['whole_bodies'] if name.startswith('blue_rail')]
        subset['visual_geoms'] = [name for name in subset['visual_geoms'] if int(name.removeprefix('visual_')) >= 107]
        root, counts = select_subset(root, subset)
    aircraft = expanded_root(xml)
    body = aircraft.find('worldbody/body')
    body.set('name', 'uav')
    body.find('freejoint').set('name', 'asm2_free')
    normal = np.asarray(target['armor_normal'], dtype=float)
    normal /= np.linalg.norm(normal)
    center = np.asarray(target['armor_surface_center_m'])
    spawn = center + normal * config['handover_standoff_m']
    body.set('pos', ' '.join(map(str, spawn)))
    root.find('worldbody').append(body)
    for section in ('asset', 'actuator', 'contact', 'sensor'):
        if (child := aircraft.find(section)) is not None:
            dest = root.find(section)
            if dest is None:
                dest = ET.SubElement(root, section)
            dest.extend(child)
    root.find('option').set('timestep', str(metadata['contact_dt_s']))
    root.find('option').set('integrator', 'implicitfast')
    scene = output / 'scene.xml'
    ET.ElementTree(root).write(scene, encoding='unicode')
    start = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(scene))
    mujoco.mj_saveModel(model, str(output / 'scene.mjb'))
    assets = {str(Path(item.attrib['file'])) for item in aircraft.findall('asset/*') if item.get('file')}
    inputs = {str(p): digest(p) for p in [registry_path, xml, metadata_path, target_path,
                                         ROOT / config['field_subset'], ROOT / 'configs/field.json', *sorted(assets)]}
    source = dict(
        field_sha256=manifest['runtime_hashes'], aircraft_and_config_sha256=inputs,
        compiled_sha256=digest(output / 'scene.mjb'), scene_xml_sha256=digest(scene),
        full_field=full_field, subset_counts=counts, subset_region_index=config['subset_region_index'],
        mujoco_version=mujoco.__version__,
        compile_wall_s=time.monotonic() - start,
    )
    write_json(output / 'scene_provenance.json', source)
    return model, metadata, target, source


class Plant:
    """Own the MuJoCo state, native named actuator bindings, and case perturbations."""

    def __init__(self, model, metadata, case):
        from rm27_contact_physics.model import VehicleEstimate

        self.estimate_type = VehicleEstimate
        self.model, self.data = model, mujoco.MjData(model)
        self.body = model.body('uav').id
        joint = model.joint('asm2_free').id
        self.qadr, self.dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.motors = np.asarray([model.actuator(f'motor_{i}').id for i in range(4)])
        self.activation = model.actuator_actadr[self.motors]
        self.nominal_mass = metadata['mass_kg']
        self.inertia = np.diag(metadata['inertia_body_kg_m2']).copy()
        self.inverse_mixer = np.linalg.inv(metadata['mixer_collective_and_body_torque_per_rotor_N'])
        self.rate_gain = metadata['rate_gain_per_s']
        self.max_motor = metadata['max_rotor_thrust_N']
        self.mass_scale = case.get('mass_scale', 1.)
        self.tau_scale = case.get('motor_tau_scale', 1.)

    def reset(self, position, velocity):
        self.data.qpos[self.qadr:self.qadr+3] = position
        self.data.qvel[self.dadr:self.dadr+3] = velocity
        hover = self.nominal_mass * 9.81 / 4
        self.data.act[self.activation] = hover
        self.data.ctrl[self.motors] = hover
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        p = self.data.qpos[self.qadr:self.qadr+7]
        v = self.data.qvel[self.dadr:self.dadr+6]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, p[3:])
        return self.estimate_type(p[:3].copy(), v[:3].copy(), rotation.reshape(3, 3),
                                  v[3:].copy(), float(self.data.time), 'MUJOCO_ORACLE')

    def command(self, action, gyro):
        from rm27_contact_physics.teacher_interface import apply_proxy

        command, clipped = apply_proxy(
            action, gyro, self.inertia, self.inverse_mixer, self.rate_gain,
            self.max_motor, return_saturation=True,
        )
        self.data.ctrl[self.motors] = command
        return clipped

    def snapshot(self):
        result = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, result, STATE)
        return result


class ContactScorer:
    """Read the completed solver interval before any endpoint recomputation."""

    def __init__(self, plant, mapping, metadata, config):
        self.model, self.body = plant.model, plant.body
        self.center = np.asarray(mapping['armor_surface_center_m'])
        self.normal = np.asarray(mapping['armor_normal'], dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.axes = np.asarray([mapping['slope_axis'], mapping['across_axis']])
        self.halfwidth = np.asarray(mapping['contact_region_halfsize_slope_across_m'])
        self.config = config
        self.patch_ids = {self.model.geom(patch['name']).id for patch in metadata['contact_patches']}
        backing = [f'rmuc2027_surface_{i}' for i in config['front_plate_backing_indices']]
        self.plate_ids = {self.model.geom(name).id for name in backing}
        self.plate_ids.update(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                              for name in mapping['registered_contact_geoms'])
        self.plate_ids.discard(-1)
        if not self.plate_ids:
            raise ValueError('Compiled scene has no registered plate geometry')
        # Derive the reference height from the compiled native pad meshes, not a sphere.
        minima = []
        for geom in self.patch_ids:
            mesh = self.model.geom_dataid[geom]
            start, length = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, self.model.geom_quat[geom])
            vertices = self.model.mesh_vert[start:start+length]
            minima.append(np.min(vertices @ matrix.reshape(3, 3)[2] + self.model.geom_pos[geom, 2]))
        self.pad_height = float(-min(minima))

    def active(self, data, before):
        from rm27_contact_physics.control import oriented_frame

        records = []
        for index, contact in enumerate(data.contact):
            first, second = (int(g) for g in contact.geom)
            if self.model.geom_bodyid[first] == self.body:
                aircraft, other, outward = first, second, -contact.frame[:3]
            elif self.model.geom_bodyid[second] == self.body:
                aircraft, other, outward = second, first, contact.frame[:3]
            else:
                continue
            if contact.efc_address < 0:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, wrench)
            if wrench[0] <= 1e-8:
                continue
            offset = contact.pos - self.center
            angular_velocity = before.rotation @ before.omega_body
            lever = contact.pos - before.position
            velocity = before.velocity + np.cross(angular_velocity, lever)
            normal_speed = point_approach_speed(before.velocity, angular_velocity, lever, np.zeros(3), self.normal)
            tangent_speed = float(np.linalg.norm(velocity + normal_speed * self.normal))
            desired_rotation = oriented_frame(self.normal, [1., 0., 0.])
            attitude_error = float(np.degrees(np.arccos(
                np.clip((np.trace(desired_rotation.T @ before.rotation)-1)/2, -1., 1.))))
            surface_error = float(np.degrees(np.arccos(np.clip(outward @ self.normal, -1., 1.))))
            c = self.config
            checks = {
                'WRONG_AIRCRAFT_GEOM': aircraft in self.patch_ids,
                'WRONG_SURFACE': other in self.plate_ids,
                'OUTSIDE_REGION': np.all(np.abs(self.axes @ offset) <= self.halfwidth),
                'PLANE_ERROR': abs(self.normal @ offset) <= c['contact_plane_error_m'],
                'SURFACE_NORMAL': surface_error <= c['contact_surface_normal_error_deg'],
                'PENETRATION': contact.dist >= -c['contact_max_penetration_m'],
                'ATTITUDE': attitude_error <= c['contact_attitude_error_deg'],
                'NORMAL_SPEED': 0 <= normal_speed <= c['contact_max_normal_speed_m_s'],
                'TANGENT_SPEED': tangent_speed <= c['contact_max_tangent_speed_m_s'],
            }
            records.append(dict(
                aircraft=self.model.geom(aircraft).name, surface=self.model.geom(other).name,
                normal_speed_m_s=normal_speed, tangent_speed_m_s=tangent_speed,
                attitude_error_deg=attitude_error, surface_normal_error_deg=surface_error,
                point_world_m=contact.pos.tolist(), force_N=float(wrench[0]),
                rejected=[name for name, passed in checks.items() if not passed],
            ))
        return records


def load_scene(config, output, existing, full_field):
    if existing is None:
        return compose_scene(config, output, full_field)
    source = json.loads((existing/'scene_provenance.json').read_text())
    if source['mujoco_version'] != mujoco.__version__:
        raise ValueError('Compiled scene MuJoCo version differs from this runtime')
    template = (existing/'scene.mjb').resolve()
    if digest(template) != source['compiled_sha256']:
        raise ValueError('Compiled scene hash mismatch')
    for path, expected in source['aircraft_and_config_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Recorded scene input changed: {path}')
    field, _, manifest = current_field()
    if manifest['runtime_hashes'] != source['field_sha256']:
        raise ValueError('Recorded field no longer matches the selected source')
    if full_field and not source['full_field']:
        raise ValueError('Expected a full-field compiled scene')
    if not source['full_field'] and source.get('subset_region_index') != config['subset_region_index']:
        raise ValueError('Compiled scene uses a different subset selection')
    (output/'scene.mjb').symlink_to(template)
    write_json(output/'scene_provenance.json', source)
    registry_path = ROOT/config['aircraft']
    registry = json.loads(registry_path.read_text())
    metadata = json.loads((registry_path.parent/registry['plant_metadata']).read_text())
    target = json.loads((ROOT/config['target_mapping']).read_text())
    return mujoco.MjModel.from_binary_path(str(template)), metadata, target, source


