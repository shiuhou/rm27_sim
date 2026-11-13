"""Independent conservative scene queries; never alter the physical model."""
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from rm27_contact_physics.control import oriented_frame


def _numbers(values):
    return ' '.join(format(float(value), '.17g') for value in np.asarray(values).ravel())


def _query_model(source, patch_names):
    """Preserve static geometry and enclose each aircraft collision category."""
    source_data = mujoco.MjData(source)
    mujoco.mj_kinematics(source, source_data)
    aircraft_body = source.body('uav').id
    root = ET.Element('mujoco')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    aircraft = ET.SubElement(world, 'body', name='uav')
    ET.SubElement(aircraft, 'freejoint', name='asm2_free')
    groups = {'query_shell': [], 'query_patch': []}
    meshes = set()
    source_vertices = {}
    for geom_id in np.flatnonzero((source.geom_contype | source.geom_conaffinity) != 0):
        name = source.geom(geom_id).name or f'geom_{geom_id}'
        is_aircraft = source.geom_bodyid[geom_id] == aircraft_body
        kind = mujoco.mjtGeom(int(source.geom_type[geom_id])).name.removeprefix('mjGEOM_').lower()
        position = source.geom_pos[geom_id] if is_aircraft else source_data.geom_xpos[geom_id]
        quaternion = (source.geom_quat[geom_id] if is_aircraft else Rotation.from_matrix(
            source_data.geom_xmat[geom_id].reshape(3, 3)).as_quat(scalar_first=True))
        attributes = dict(name=name, type=kind, pos=_numbers(position), quat=_numbers(quaternion),
                          contype='1' if is_aircraft else '2', conaffinity='2' if is_aircraft else '1')
        if kind == 'mesh':
            mesh_id = int(source.geom_dataid[geom_id])
            first = source.mesh_vertadr[mesh_id]
            vertices = source.mesh_vert[first:first+source.mesh_vertnum[mesh_id]]
            if is_aircraft:
                group = 'query_patch' if name in patch_names else 'query_shell'
                groups[group].append(Rotation.from_quat(quaternion, scalar_first=True).apply(vertices)+position)
                continue
            mesh_name = f'query_mesh_{mesh_id}'
            if mesh_name not in meshes:
                ET.SubElement(asset, 'mesh', name=mesh_name, vertex=_numbers(vertices))
                meshes.add(mesh_name)
            attributes['mesh'] = mesh_name
        else:
            attributes['size'] = _numbers(source.geom_size[geom_id])
        ET.SubElement(aircraft if is_aircraft else world, 'geom', **attributes)
    for group, parts in groups.items():
        if not parts:
            continue
        vertices = np.concatenate(parts)
        hull = ConvexHull(vertices)
        source_vertices[group] = dict(vertices=int(len(vertices)), hull_vertices=int(len(hull.vertices)),
                                      maximum_outside_m=float(np.max(vertices @ hull.equations[:, :3].T
                                                                    + hull.equations[:, 3])))
        ET.SubElement(asset, 'mesh', name=group, vertex=_numbers(vertices[hull.vertices]))
        ET.SubElement(aircraft, 'geom', name=group, type='mesh', mesh=group, contype='1', conaffinity='2')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode')), source_vertices


class SceneClearance:
    def __init__(self, model_path, patch_names, plate_names):
        source = mujoco.MjModel.from_binary_path(str(model_path))
        self.model, self.enclosure = _query_model(source, set(patch_names))
        self.data = mujoco.MjData(self.model)
        self.patch_names, self.plate_names = set(patch_names) | {'query_patch'}, set(plate_names)
        self.body = self.model.body('uav').id
        self.qadr = int(self.model.jnt_qposadr[self.model.joint('asm2_free').id])

    def check(self, reference, now, deadline, dt=.02, end_at=None):
        """Sample conservative hulls against every enabled static scene geometry."""
        end = (reference.end if reference.contact_preview is None else
               min(reference.end, reference.contact_preview['earliest_window_s']+.005))
        end = min(end, end_at) if end_at is not None else end
        times = np.linspace(now, end, max(2, int(np.ceil((end-now)/dt))+1))
        positions, _, accelerations, _ = reference.samples(times)
        rotations = np.array([oriented_frame(force, [1., 0., 0.])
                              for force in accelerations+[0., 0., 9.81]])
        contact_time = (np.inf if reference.contact_preview is None else
                        reference.contact_preview['earliest_window_s'])
        for timestamp, position, rotation in zip(times, positions, rotations):
            if time.perf_counter() >= deadline:
                return False, 'SOLVER_DEADLINE'
            self.data.qpos[self.qadr:self.qadr+3] = position
            self.data.qpos[self.qadr+3:self.qadr+7] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_collision(self.model, self.data)
            for contact in self.data.contact:
                geom_ids = list(contact.geom)
                if not any(self.model.geom_bodyid[geom_id] == self.body for geom_id in geom_ids):
                    continue
                names = [self.model.geom(geom_id).name for geom_id in geom_ids]
                aircraft = next(names[index] for index, geom_id in enumerate(geom_ids)
                                if self.model.geom_bodyid[geom_id] == self.body)
                surface = next((names[index] for index, geom_id in enumerate(geom_ids)
                                if self.model.geom_bodyid[geom_id] != self.body), '')
                allowed = (timestamp >= contact_time and aircraft in self.patch_names
                           and surface in self.plate_names)
                if contact.dist <= 0 and not allowed:
                    return False, 'PREDICTED_BODY_OR_EARLY_CONTACT'
        return True, None
