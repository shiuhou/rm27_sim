"""Shared contact/recovery scoring on completed MuJoCo physics intervals.

The implicitfast solve left by mj_step belongs to the integrated interval.
Capture it before mj_forward refreshes the endpoint contact snapshot. Snapshot
queries never integrate force or advance time; a timestamp is consumed once.
"""
from __future__ import annotations

import copy

import numpy as np

from rm27_contact_physics.contact_metrics import armor_contacts, point_approach_speed
from .model import advance_physics_interval, target_frame, truth_estimate, unit


class ContactEvaluator:
    """Own task history, independently of controllers and recording."""

    protocol = "completed_interval_implicitfast_v1"

    def __init__(self, cfg, physical, episode_start=0.):
        self.g = copy.deepcopy(cfg["gates"])
        self.physical = copy.deepcopy(physical)
        self.episode_start = float(episode_start)
        self.normal = unit(physical["target"]["normal"])
        self.target_R = target_frame(physical["target"])
        self.center = np.array(physical["target"]["center_m"])
        self.half = np.array(physical["target"]["dimensions_m"][:2]) / 2
        self.first = None
        self.reasons = set()
        self.events = []
        self.last_touch = None
        self.secondary = False
        self.contact_latch = False
        self.touching_count = 0
        self.separated = False
        self.recovery_time = self.hold_start = self.recovery_distance = None
        self.contact_height = None
        self.post_min, self.post_max = np.inf, -np.inf
        self.max_rate = self.max_speed = self.peak_force = self.impulse = 0.
        self.sat_run = self.max_sat_run = self.saturated_time = self.elapsed_time = 0.
        self.min_distance = self.min_altitude = np.inf
        self.max_tilt = self.max_flight_speed = 0.
        self.completed_intervals = 0
        self.interval_end_time = self.episode_start
        self.snapshot_time = None
        self._final_result = None

    @property
    def recovered(self):
        return self.recovery_time is not None

    def initialize(self, model, data, state=None):
        if self.snapshot_time is not None:
            raise RuntimeError("Evaluator already initialized")
        state = truth_estimate(model, data) if state is None else state
        if not np.isclose(state.timestamp, self.episode_start, rtol=0, atol=1e-12):
            raise ValueError("Initial snapshot time differs from episode start")
        self.snapshot(model, data, state, state)

    def snapshot(self, model, data, state, pre_state):
        """Consume one forwarded boundary; repeated refreshes are read-only."""
        if self._final_result is not None:
            return
        t = float(state.timestamp)
        if self.snapshot_time is not None:
            if np.isclose(t, self.snapshot_time, rtol=0, atol=1e-12):
                return
            if t < self.snapshot_time:
                raise ValueError("Snapshot time moved backwards")
        if not np.isclose(t, self.interval_end_time, rtol=0, atol=1e-12):
            raise ValueError("Snapshot requires a completed interval at this time")
        self.snapshot_time = t
        if not np.isfinite(np.r_[state.position, state.velocity, state.rotation.ravel(), state.omega_body]).all():
            self.reasons.add("NUMERICAL_FAILURE")
            return
        g = self.g
        self.max_tilt = max(self.max_tilt, float(np.rad2deg(np.arccos(np.clip(state.rotation[2, 2], -1, 1)))))
        self.max_flight_speed = max(self.max_flight_speed, float(np.linalg.norm(state.velocity)))
        self.min_altitude = min(self.min_altitude, float(state.position[2]))
        armor = model.geom("armor").id
        touching = []
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 0 or c.efc_address < 0:
                continue
            names = [model.geom(int(v)).name for v in c.geom]
            if (model.body("uav").id in model.geom_bodyid[c.geom] and "armor" not in names
                    and not any(n.startswith("rim_") for n in names) and "ground" not in names):
                self.reasons.add("ENVIRONMENT_CONTACT")
            if "ground" in names:
                self.reasons.add("GROUND_CONTACT")
            if "armor" in names or any(n.startswith("rim_") for n in names):
                touching.append((c, names))
        self.touching_count = len(touching)
        self.contact_latch |= bool(touching)
        pc = state.position + state.rotation @ self.physical["contact"]["offset_body_m"]
        self.min_distance = min(self.min_distance, float(np.linalg.norm(pc - self.center)))
        if touching:
            if self.last_touch is not None and t - self.last_touch > g["secondary_contact_gap_s"]:
                self.secondary = True
            self.last_touch = t
            if self.first is None:
                valid_part = all(set(names) == {"armor", "bumper"} for _, names in touching)
                c = touching[0][0]
                local = self.target_R.T @ (c.pos - self.center)
                r_world = np.asarray(c.pos) - pre_state.position
                omega_world = pre_state.rotation @ pre_state.omega_body
                point_velocity = pre_state.velocity + np.cross(omega_world, r_world)
                vn = point_approach_speed(pre_state.velocity, omega_world, r_world, np.zeros(3), self.normal)
                vt = float(np.linalg.norm(point_velocity + vn * self.normal))
                angle = float(np.rad2deg(np.arccos(np.clip(pre_state.rotation[:, 2] @ self.normal, -1, 1))))
                edge = bool(np.any(np.abs(local[:2]) > self.half - g["edge_margin_m"]))
                outward = c.frame.reshape(3, 3)[0] * (1 if c.geom[0] == armor else -1)
                front_face = bool(float(outward @ self.normal) > .9)
                self.first = dict(time_s=t, pairs=[names for _, names in touching],
                                  valid_part=valid_part, edge=edge, front_face=front_face,
                                  position_world_m=c.pos.tolist(), position_error_m=float(np.linalg.norm(local[:2])),
                                  normal_speed_m_s=vn, tangent_speed_m_s=vt, attitude_error_deg=angle,
                                  body_rate_rad_s=float(np.linalg.norm(pre_state.omega_body)),
                                  region_margin_m=float(np.min(self.half - np.abs(local[:2]))),
                                  local_coordinates_m=local.tolist(), motor_state_N=data.act.tolist(),
                                  pre_velocity_world_m_s=pre_state.velocity.tolist(),
                                  pre_omega_body_rad_s=pre_state.omega_body.tolist(),
                                  pre_position_world_m=pre_state.position.tolist(),
                                  pre_rotation_body_to_world=pre_state.rotation.tolist(),
                                  pre_state_timestamp_s=pre_state.timestamp)
                for invalid, reason in ((not valid_part, "WRONG_FIRST_CONTACT"), (edge, "EDGE_CONTACT"),
                                        (not front_face, "WRONG_TARGET_FACE"),
                                        (vt > g["max_contact_tangent_speed_m_s"], "EXCESSIVE_TANGENTIAL_SLIP"),
                                        (not g["contact_normal_speed_range_m_s"][0] <= vn <= g["contact_normal_speed_range_m_s"][1], "CONTACT_SPEED_OUT_OF_RANGE"),
                                        (angle > g["max_contact_attitude_error_deg"], "CONTACT_ATTITUDE_ERROR")):
                    if invalid:
                        self.reasons.add(reason)
                self.contact_height = float(state.position[2])
                self.events.append(dict(event="FIRST_CONTACT", **copy.deepcopy(self.first)))
            if any(set(names) != {"armor", "bumper"} for _, names in touching):
                self.reasons.add("UNDESIGNATED_CONTACT")
            force = sum(r["compressive_normal_force_N"] for r in armor_contacts(model, data, armor, self.normal))
            self.peak_force = max(self.peak_force, force)
        if self.first is not None:
            self.post_min = min(self.post_min, float(state.position[2]))
            self.post_max = max(self.post_max, float(state.position[2]))
            self.max_rate = max(self.max_rate, float(np.linalg.norm(state.omega_body)))
            self.max_speed = max(self.max_speed, float(np.linalg.norm(state.velocity)))
            clearance = float((pc - self.center) @ self.normal - self.physical["contact"]["radius_m"])
            if clearance > g["separation_m"] and not touching and not self.separated:
                self.separated = True
                self.events.append(dict(time_s=t, event="SEPARATED"))
            tilt = np.rad2deg(np.arccos(np.clip(state.rotation[2, 2], -1, 1)))
            stable = (clearance > g["separation_m"] and not touching and tilt <= g["max_recovery_tilt_deg"]
                      and np.linalg.norm(state.omega_body) <= g["max_recovery_rate_rad_s"]
                      and np.linalg.norm(state.velocity) <= g["max_recovery_speed_m_s"])
            self.hold_start = (t if self.hold_start is None else self.hold_start) if stable else None
            if (self.hold_start is not None and t - self.hold_start >= g["recovery_hold_s"] - 1e-12
                    and self.recovery_time is None):
                self.recovery_time = self.hold_start - self.first["time_s"]
                self.recovery_distance = float(np.linalg.norm(state.position - self.first["position_world_m"]))
                self.events.append(dict(time_s=t, event="RECOVERED", stable_since_s=self.hold_start))
            if np.any(state.position < g["recovery_volume_min_m"]) or np.any(state.position > g["recovery_volume_max_m"]):
                self.reasons.add("RECOVERY_VOLUME_EXIT")

    def integrate_interval(self, model, data, start_time, saturated=False):
        """Consume the implicitfast solve immediately after a completed step."""
        if self._final_result is not None:
            raise RuntimeError("Evaluator has a finalized result")
        start, end = float(start_time), float(data.time)
        if (self.snapshot_time is None or not np.isclose(start, self.snapshot_time, rtol=0, atol=1e-12)
                or not np.isclose(start, self.interval_end_time, rtol=0, atol=1e-12)):
            raise ValueError("Interval start must match the last consumed boundary")
        dt = end - start
        if not np.isclose(dt, model.opt.timestep, rtol=0, atol=1e-12) or dt <= 0:
            raise ValueError("Expected exactly one completed physics interval")
        force = sum(r["compressive_normal_force_N"] for r in armor_contacts(model, data, model.geom("armor").id, self.normal))
        self.impulse += force * dt
        self.elapsed_time += dt
        self.saturated_time += dt if saturated else 0.
        self.sat_run = self.sat_run + dt if saturated else 0.
        self.max_sat_run = max(self.max_sat_run, self.sat_run)
        self.completed_intervals += 1
        self.interval_end_time = end

    def advance(self, model, data, before, saturated=False):
        """One physics step, interval force capture, then endpoint sampling."""
        if self._final_result is not None:
            raise RuntimeError("Evaluator has a finalized result")
        if (self.snapshot_time is None
                or not np.isclose(data.time, before.timestamp, rtol=0, atol=1e-12)
                or not np.isclose(before.timestamp, self.snapshot_time, rtol=0, atol=1e-12)
                or not np.isclose(before.timestamp, self.interval_end_time, rtol=0, atol=1e-12)):
            raise ValueError("Pre-contact state must be the current interval start")
        state = advance_physics_interval(model, data, before,
                                         consume_interval=self.integrate_interval,
                                         saturated=saturated)
        self.snapshot(model, data, state, before)
        return state

    def consume_contact(self):
        contact = self.contact_latch
        self.contact_latch = False
        return contact

    def irreversible_reasons(self, model, data):
        if self._final_result is not None:
            return list(self._final_result["failure_modes"])
        reasons = set(self.reasons)
        if self.secondary:
            reasons.add("SECONDARY_CONTACT")
        if self.first is not None:
            if self.max_rate > self.g["max_body_rate_rad_s"]:
                reasons.add("EXCESSIVE_BODY_RATE")
            if self.contact_height - self.post_min > self.g["max_altitude_loss_m"]:
                reasons.add("ALTITUDE_LOSS")
            if self.recovery_time is not None and self.recovery_time > self.g["recovery_deadline_s"]:
                reasons.add("RECOVERY_TIMEOUT")
        if self.max_sat_run > self.g["max_saturation_run_s"]:
            reasons.add("ACTUATOR_SATURATION")
        if any(data.warning[i].number for i in range(len(data.warning))):
            reasons.add("SIMULATOR_WARNING")
        self.reasons.update(reasons)
        return sorted(reasons)

    def summary(self, model, data):
        if self._final_result is not None:
            return copy.deepcopy(self._final_result)
        reasons = self.irreversible_reasons(model, data)
        valid = bool(self.first and self.first["valid_part"] and self.first["front_face"] and not self.first["edge"])
        return dict(passed=bool(self.recovered and not reasons), first_contact=copy.deepcopy(self.first),
                    valid_contact=valid, separated=self.separated, secondary_contact=self.secondary,
                    recovery_time_s=self.recovery_time, recovery_distance_m=self.recovery_distance,
                    max_tilt_deg=self.max_tilt, max_flight_speed_m_s=self.max_flight_speed,
                    min_altitude_m=float(self.min_altitude), max_post_body_rate_rad_s=self.max_rate,
                    max_post_speed_m_s=self.max_speed, min_post_altitude_m=None if self.first is None else float(self.post_min),
                    max_post_altitude_m=None if self.first is None else float(self.post_max),
                    altitude_loss_m=None if self.first is None else float(max(0., self.contact_height - self.post_min)),
                    peak_normal_force_N=self.peak_force, normal_impulse_Ns=self.impulse,
                    saturation_fraction=self.saturated_time / self.elapsed_time if self.elapsed_time else 0.,
                    max_saturation_run_s=self.max_sat_run, closest_bumper_center_m=float(self.min_distance),
                    failure_modes=reasons, duration_s=float(data.time), episode_elapsed_s=float(data.time - self.episode_start),
                    referee_scoring_success=None, events=copy.deepcopy(self.events),
                    evaluation_protocol=self.protocol, completed_physics_intervals=self.completed_intervals)

    def finalize(self, model, data):
        if self._final_result is None:
            self.irreversible_reasons(model, data)
            if self.first is None:
                self.reasons.add("TARGET_MISS")
            else:
                if not self.separated:
                    self.reasons.add("NO_SEPARATION")
                if self.recovery_time is None or self.recovery_time > self.g["recovery_deadline_s"]:
                    self.reasons.add("RECOVERY_TIMEOUT")
            result = self.summary(model, data)
            result["passed"] = not self.reasons
            self._final_result = result
        return copy.deepcopy(self._final_result)
