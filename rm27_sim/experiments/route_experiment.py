"""Fast red-dock to blue-armor contact and return experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rm27_sim.paths import ROOT, resolve_asset


def path_distance(position, points):
    p = np.asarray(position)
    points = np.asarray(points)
    a, b = points[:-1], points[1:]
    v = b-a
    denominator = np.einsum('ij,ij->i', v, v)
    t = np.divide(np.einsum('ij,ij->i', p-a, v), denominator,
                  out=np.zeros(len(v)), where=denominator > 0)
    closest = a + np.clip(t, 0, 1)[:, None]*v
    return float(np.linalg.norm(closest-p, axis=1).min())


class RouteExperiment:
    """Follow an audited corridor, touch the enemy armor, and return."""

    def __init__(self, scene, audit_path, mission_path=ROOT/'configs/fast_armor_mission.json'):
        scene = Path(scene)
        self.audit = json.loads(Path(audit_path).read_text())
        self.mission = json.loads(Path(mission_path).read_text())
        if self.mission.get('friendly_team') != 'red' or self.mission.get('target_team') != 'blue':
            raise ValueError('Route must launch red and target blue')
        if self.audit.get('passed') is not True or not self.audit.get('surface_route_clear'):
            raise ValueError('Route surface audit is not clear')
        if hashlib.sha256(scene.read_bytes()).hexdigest() != self.audit['scene_sha256']:
            raise ValueError('Route audit scene hash mismatch')
        if 'asset_sha256' not in self.audit:
            raise ValueError('Rebuild route audit with asset hashes')
        for path, digest in self.audit['asset_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Route asset hash mismatch: {path}')
        self.clearance_points = np.asarray(self.audit['waypoints_world_m'], dtype=float)
        if (self.clearance_points.ndim != 2 or self.clearance_points.shape[1] != 3
                or len(self.clearance_points) < 2 or not np.isfinite(self.clearance_points).all()):
            raise ValueError('Invalid audited waypoints')
        indices = self.mission['command_waypoint_indices']
        if indices[0] != 0 or indices[-1] != len(self.clearance_points)-1 or indices != sorted(set(indices)):
            raise ValueError('Invalid command waypoint indices')
        self.points = self.clearance_points[indices]
        self.stop_waypoints = set(self.mission['stop_waypoint_indices'])
        self.inbound_stop_waypoints = set(self.mission['inbound_stop_waypoint_indices'])
        valid_waypoints = set(range(1, len(self.points)))
        if (not self.stop_waypoints <= valid_waypoints
                or not self.inbound_stop_waypoints <= valid_waypoints):
            raise ValueError('Invalid stop waypoint indices')
        self.margin = float(self.audit['tracking_margin_m'])
        if not 0 < self.margin <= .1:
            raise ValueError('Expected a positive audited tracking margin at most .1 m')

        model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom_name, site_name = self.mission['armor_geom'], self.mission['armor_site']
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) < 0:
            raise ValueError(f'Missing enemy armor geom: {geom_name}')
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f'Missing enemy armor site: {site_name}')
        self.armor_center = data.site_xpos[site_id].copy()
        self.armor_normal = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
        expected_staging = self.armor_center + self.armor_normal*self.mission['staging_distance_m']
        if not np.allclose(self.points[-1], expected_staging, atol=.002, rtol=0):
            raise ValueError('Audited route does not end outside the blue armor')
        if self.armor_center[0] >= 0 or self.points[0][0] <= 0:
            raise ValueError('Expected red launch on +X side and blue armor on -X side')

        self.state = 'OUTBOUND'
        self.index = 1
        self.reference = self.points[0].copy()
        self.reference_velocity = np.zeros(3)
        self.last_time = None
        self.max_path_error = 0.
        self.visited = []
        self.impact = None
        self.physics_deadline = 100.
        self.maximum_speed_m_s = float(self.mission['maximum_flight_speed_m_s'])
        self.maximum_tilt_deg = float(self.mission['maximum_tilt_deg'])
        self.horizontal_command_limit_m_s = float(self.mission['horizontal_command_limit_m_s'])
        self.vertical_command_limit_m_s = float(self.mission['vertical_command_limit_m_s'])

    def _move_reference(self, target, rate, dt, *, brake=True):
        delta = np.asarray(target)-self.reference
        distance = np.linalg.norm(delta)
        if distance:
            if brake:
                vertical = abs(delta[2]) > np.linalg.norm(delta[:2])
                acceleration = self.mission['vertical_reference_braking_m_s2' if vertical
                                            else 'horizontal_reference_braking_m_s2']
                effective_rate = min(rate, float(np.sqrt(2*acceleration*distance)))
            else:
                effective_rate = rate
            self.reference_velocity = delta/distance*effective_rate
            self.reference += delta*min(1., effective_rate*dt/distance)
            if effective_rate*dt >= distance:
                self.reference_velocity[:] = 0
        else:
            self.reference_velocity[:] = 0
        return self.reference.copy()

    def _segment_rate(self, target, nominal):
        delta = np.asarray(target)-self.reference
        if abs(delta[2]) > np.linalg.norm(delta[:2]):
            return min(nominal, self.mission['vertical_reference_speed_m_s'])
        return nominal

    def _record(self, time, position, name):
        self.visited.append({'state': name, 'time_s': float(time),
                             'position': np.asarray(position).tolist()})

    def check(self, position):
        if self.state in ('IMPACT', 'RETREAT'):
            path = np.vstack([self.points[-1], self.armor_center])
        elif self.state == 'OUTBOUND' and self.index == len(self.points)-1:
            path = np.vstack([self.clearance_points, self.armor_center])
        else:
            path = self.clearance_points
        error = path_distance(position, path)
        self.max_path_error = max(error, self.max_path_error)
        # The fast altitude transition can overshoot upward while remaining over
        # the audited XY centerline. Upward-only allowance is bounded separately;
        # lateral and downward deviations retain the audited tube limit.
        horizontal_error = path_distance(np.r_[position[:2], 0],
                                         np.c_[path[:, :2], np.zeros(len(path))])
        upward_overshoot = float(position[2]-np.max(path[:, 2]))
        bounded_upward = (horizontal_error <= self.margin and 0 < upward_overshoot
                          <= self.mission['maximum_high_altitude_overshoot_m'])
        if error > self.mission['runtime_tracking_abort_m'] and not bounded_upward:
            raise RuntimeError(f'Route tracking tube exceeded: {error:.6f} m')

    def contact_allowed(self, contact):
        return self.mission['armor_geom'] in contact['geoms'] and self.state in ('IMPACT', 'RETREAT')

    def target(self, time, position, speed, contacts=(), velocity=None):
        position = np.asarray(position)
        velocity = np.zeros(3) if velocity is None else np.asarray(velocity)
        dt = 0. if self.last_time is None else max(0., time-self.last_time)
        self.last_time = time
        waypoint_radius = self.mission['waypoint_radius_m']

        if self.state == 'OUTBOUND':
            target = self.points[self.index]
            final_waypoint = self.index == len(self.points)-1
            stop_waypoint = self.index in self.stop_waypoints
            # Advance intermediate points when the commanded reference reaches
            # them. Waiting for the aircraft would stop the reference at every
            # point and turn a smooth arc into a series of abrupt corners.
            if (not final_waypoint and not stop_waypoint
                    and np.linalg.norm(self.reference-target) < 1e-9):
                self._record(time, position, f'outbound_reference_{self.index}')
                self.index += 1
                target = self.points[self.index]
                final_waypoint = self.index == len(self.points)-1
            if (stop_waypoint and np.linalg.norm(position-target) < waypoint_radius
                    and speed < self.mission['waypoint_speed_gate_m_s']):
                self._record(time, position, f'outbound_{self.index}')
                self.index += 1
                if self.index == len(self.points):
                    self.state = 'IMPACT'
                    self.reference = position.copy()
                    target = self.armor_center-self.armor_normal*.05
                else:
                    target = self.points[self.index]
            rate = self._segment_rate(target, self.mission['cruise_reference_speed_m_s'])
            return self._move_reference(target, rate, dt, brake=stop_waypoint)

        if self.state == 'IMPACT':
            for contact in contacts:
                if self.mission['armor_geom'] in contact['geoms']:
                    normal_speed = -float(velocity @ self.armor_normal)
                    self.impact = {
                        'time_s': float(time), 'position_world_m': position.tolist(),
                        'velocity_world_m_s': velocity.tolist(),
                        'armor_normal_speed_m_s': normal_speed,
                        'contact': contact,
                    }
                    self._record(time, position, 'armor_contact')
                    self.state = 'RETREAT'
                    self.reference = position.copy()
                    break
            target = self.points[-1] if self.state == 'RETREAT' else self.armor_center-self.armor_normal*.05
            rate = (self.mission['retreat_reference_speed_m_s'] if self.state == 'RETREAT'
                    else self.mission['impact_reference_speed_m_s'])
            return self._move_reference(target, self._segment_rate(target, rate), dt)

        if self.state == 'RETREAT':
            target = self.points[-1]
            if (np.linalg.norm(position-target) < waypoint_radius
                    and speed < self.mission['waypoint_speed_gate_m_s']):
                self._record(time, position, 'retreated')
                self.state = 'INBOUND'
                self.index = len(self.points)-2
                self.reference = position.copy()
                target = self.points[self.index]
            rate = self._segment_rate(target, self.mission['retreat_reference_speed_m_s'])
            return self._move_reference(target, rate, dt)

        target = self.points[self.index]
        stop_waypoint = self.index in self.inbound_stop_waypoints
        if (self.index > 0 and not stop_waypoint
                and np.linalg.norm(self.reference-target) < 1e-9):
            self._record(time, position, f'inbound_reference_{self.index}')
            self.index -= 1
            target = self.points[self.index]
        radius = self.mission['return_radius_m'] if self.index == 0 else waypoint_radius
        if np.linalg.norm(position-target) < radius:
            if self.index == 0:
                if speed <= self.mission['return_speed_gate_m_s']:
                    self._record(time, position, 'returned_over_dock')
                    return None
            elif (not stop_waypoint
                  or speed < self.mission['waypoint_speed_gate_m_s']):
                self._record(time, position, f'inbound_{self.index}')
                self.index -= 1
                target = self.points[self.index]
        rate = self._segment_rate(target, self.mission['cruise_reference_speed_m_s'])
        return self._move_reference(target, rate, dt,
                                    brake=self.index == 0 or stop_waypoint)


if __name__ == '__main__':
    from ..autopilot.micro_flight import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--mission', type=Path, default=ROOT/'configs/fast_armor_mission.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance', type=int, default=43)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speedup', type=float)
    args = parser.parse_args()
    route = RouteExperiment(args.scene, args.audit, args.mission)
    raise SystemExit(run(args.scene, args.output, args.instance, route=route, view=args.view, speedup=args.speedup))
