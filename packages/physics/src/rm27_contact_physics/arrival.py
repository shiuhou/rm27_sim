"""Independent no-contact terminal-arrival scoring on the shared plant.

The task uses state truth for evaluation, not policy inputs. A facing geometric
point and the instantaneous material point occupying it have different velocity
semantics. No reward or learning framework belongs to this module.
"""
from __future__ import annotations

import copy

import numpy as np

from .model import advance_physics_interval, frame, truth_estimate, unit


def terminal_geometry(task_config, physical):
    """Return world-frame terminal geometry from explicit configuration.

    ``target.center_m`` is the armor front-face center. The sphere-to-face gap
    already measures from that surface; half the plate thickness is not added.
    """
    terminal = task_config["terminal"]
    center = np.asarray(physical["target"]["center_m"], dtype=float)
    normal = unit(physical["target"]["normal"])
    offset = np.asarray(physical["contact"]["offset_body_m"], dtype=float)
    radius = float(physical["contact"]["radius_m"])
    gap = float(terminal["gap_m"])
    speed = float(terminal["normal_speed_m_s"])
    if (center.shape != (3,) or offset.shape != (3,)
            or not np.isfinite(np.r_[center, offset, radius, gap, speed]).all()
            or radius <= 0 or gap <= 0 or speed <= 0):
        raise ValueError("Expected finite terminal geometry and positive radius, gap and speed")
    rotation = frame(normal)
    bumper_target = center + (gap + radius) * normal
    return dict(center=center, normal=normal, target_rotation=rotation,
                bumper_target=bumper_target,
                position_target=bumper_target - rotation @ offset,
                velocity_target=-speed * normal, offset_body=offset, radius=radius)


class ArrivalEvaluator:
    """Own terminal/safety history; ContactEvaluator remains a separate task."""

    protocol = "completed_interval_implicitfast_v1"

    def __init__(self, task_config, physical, episode_start=0.):
        self.task = copy.deepcopy(task_config)
        self.physical = copy.deepcopy(physical)
        self.geometry = terminal_geometry(self.task, self.physical)
        self.terminal = self.task["terminal"]
        self.safety = self.task["safety"]
        self.horizon_s = float(self.task["horizon_s"])
        self.episode_start = float(episode_start)
        positive = [self.horizon_s]
        positive += [float(self.terminal[key]) for key in (
            "gap_tolerance_m", "tangent_position_tolerance_m",
            "normal_speed_tolerance_m_s", "tangent_speed_tolerance_m_s",
            "attitude_tolerance_deg", "body_rate_tolerance_rad_s")]
        positive += [float(self.safety[key]) for key in (
            "max_speed_m_s", "max_tilt_deg", "max_body_rate_rad_s",
            "max_saturation_run_s", "min_gap_m")]
        self.position_min = np.asarray(self.safety["position_min_m"], dtype=float)
        self.position_max = np.asarray(self.safety["position_max_m"], dtype=float)
        if (not np.isfinite(positive).all() or any(value <= 0 for value in positive)
                or not np.isfinite(self.episode_start)
                or self.position_min.shape != (3,) or self.position_max.shape != (3,)
                or not np.isfinite(np.r_[self.position_min, self.position_max]).all()
                or np.any(self.position_min >= self.position_max)):
            raise ValueError("Arrival task requires positive finite limits and an ordered safety box")
        self.reasons = set()
        self.events = []
        self.arrival_time = None
        self.snapshot_time = None
        self.interval_end_time = self.episode_start
        self.completed_intervals = 0
        self.elapsed_time = self.saturated_time = self.sat_run = self.max_sat_run = 0.
        self.path_length = 0.
        self.min_plane_gap = np.inf
        self.max_speed = self.max_tilt = self.max_body_rate = 0.
        self.last_metrics = None
        self.best_violation_ratio = np.inf
        self._last_position = None
        self._final_result = None

    @property
    def arrived(self):
        return self.arrival_time is not None

    @property
    def done(self):
        return bool(self.arrived or self.reasons)

    def metrics(self, state):
        """Measure all terminal conditions at one state timestamp, in SI units."""
        geometry = self.geometry
        normal, center = geometry["normal"], geometry["center"]
        offset_world = state.rotation @ geometry["offset_body"]
        omega_world = state.rotation @ state.omega_body
        bumper = state.position + offset_world
        facing = bumper - geometry["radius"] * normal
        bumper_velocity = state.velocity + np.cross(omega_world, offset_world)
        material_velocity = state.velocity + np.cross(
            omega_world, offset_world - geometry["radius"] * normal)
        gap = float(normal @ (facing - center))
        tangent_error = float(np.linalg.norm(facing - center - gap * normal))
        com_normal = -float(normal @ state.velocity)
        material_normal = -float(normal @ material_velocity)
        com_tangent = float(np.linalg.norm(state.velocity + com_normal * normal))
        material_tangent = float(np.linalg.norm(material_velocity + material_normal * normal))
        attitude = float(np.arccos(np.clip(
            (np.trace(geometry["target_rotation"].T @ state.rotation) - 1.) / 2., -1., 1.)))
        return dict(bumper_center=bumper, facing_surface_point=facing,
                    bumper_center_velocity=bumper_velocity,
                    material_velocity_at_facing_point=material_velocity,
                    plane_gap_m=gap, plane_gap_error_m=abs(gap - self.terminal["gap_m"]),
                    tangent_position_error_m=tangent_error,
                    position_error_m=float(np.linalg.norm(state.position - geometry["position_target"])),
                    bumper_position_error_m=float(np.linalg.norm(bumper - geometry["bumper_target"])),
                    com_normal_speed_m_s=com_normal, material_normal_speed_m_s=material_normal,
                    com_tangent_speed_m_s=com_tangent, material_tangent_speed_m_s=material_tangent,
                    velocity_error_m_s=max(float(np.linalg.norm(state.velocity - geometry["velocity_target"])),
                                          float(np.linalg.norm(material_velocity - geometry["velocity_target"]))),
                    attitude_error_rad=attitude, attitude_error_deg=float(np.rad2deg(attitude)),
                    angular_velocity_error_rad_s=float(np.linalg.norm(state.omega_body)),
                    flight_speed_m_s=float(np.linalg.norm(state.velocity)),
                    tilt_deg=float(np.rad2deg(np.arccos(np.clip(state.rotation[2, 2], -1., 1.)))))

    def terminal_normalized_errors(self, metrics):
        """Each terminal error over its tolerance, in the order of the arrival gates."""
        terminal = self.terminal
        speed = terminal["normal_speed_m_s"]
        return np.array([
            metrics["plane_gap_error_m"] / terminal["gap_tolerance_m"],
            metrics["tangent_position_error_m"] / terminal["tangent_position_tolerance_m"],
            abs(metrics["com_normal_speed_m_s"] - speed) / terminal["normal_speed_tolerance_m_s"],
            abs(metrics["material_normal_speed_m_s"] - speed) / terminal["normal_speed_tolerance_m_s"],
            metrics["com_tangent_speed_m_s"] / terminal["tangent_speed_tolerance_m_s"],
            metrics["material_tangent_speed_m_s"] / terminal["tangent_speed_tolerance_m_s"],
            metrics["attitude_error_deg"] / terminal["attitude_tolerance_deg"],
            metrics["angular_velocity_error_rad_s"] / terminal["body_rate_tolerance_rad_s"]])

    def terminal_violation_ratio(self, metrics):
        """Largest terminal error over its tolerance; <= 1 exactly when all gates qualify.

        Diagnostic and shaping input only: the arrival decision in ``snapshot``
        keeps its own direct comparisons.
        """
        return float(self.terminal_normalized_errors(metrics).max())

    def _fail(self, reason):
        self.reasons.add(reason)

    def _external_contacts(self, model, data, *, sampling, interval_start=None):
        """Latch valid UAV/environment collisions in either sampling stage."""
        uav = model.body("uav").id
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.dist > 0 or contact.efc_address < 0:
                continue
            body_ids = model.geom_bodyid[contact.geom]
            if np.count_nonzero(body_ids == uav) != 1:
                continue
            names = [model.geom(int(geom)).name for geom in contact.geom]
            reason = ("GROUND_CONTACT" if "ground" in names else
                      "ARMOR_CONTACT" if "armor" in names else
                      "RIM_CONTACT" if any(name.startswith("rim_") for name in names) else
                      "ENVIRONMENT_CONTACT")
            if reason not in self.reasons:
                event = dict(event="SAFETY_CONTACT", failure_mode=reason,
                             pairs=[names], sampling_source=sampling)
                if interval_start is None:
                    event["time_s"] = float(data.time)
                else:
                    event.update(interval_start_s=float(interval_start), interval_end_s=float(data.time))
                self.events.append(event)
            self._fail(reason)

    def initialize(self, model, data, state=None):
        if self.snapshot_time is not None:
            raise RuntimeError("Evaluator already initialized")
        state = truth_estimate(model, data) if state is None else state
        if not np.isclose(state.timestamp, self.episode_start, rtol=0, atol=1e-12):
            raise ValueError("Initial snapshot time differs from episode start")
        self.snapshot(model, data, state)

    def snapshot(self, model, data, state):
        """Consume the forwarded endpoint once; a refresh cannot accrue time."""
        if self._final_result is not None:
            return
        timestamp = float(state.timestamp)
        if self.snapshot_time is not None:
            if np.isclose(timestamp, self.snapshot_time, rtol=0, atol=1e-12):
                return
            if timestamp < self.snapshot_time:
                raise ValueError("Snapshot time moved backwards")
        if not np.isclose(timestamp, self.interval_end_time, rtol=0, atol=1e-12):
            raise ValueError("Snapshot requires a completed interval at this time")
        self.snapshot_time = timestamp
        self._external_contacts(model, data, sampling="FORWARDED_ENDPOINT")
        if not np.isfinite(np.r_[state.position, state.velocity, state.rotation.ravel(), state.omega_body]).all():
            self._fail("NUMERICAL_FAILURE")
            return
        metrics = self.last_metrics = self.metrics(state)
        if self._last_position is not None:
            self.path_length += float(np.linalg.norm(state.position - self._last_position))
        self._last_position = state.position.copy()
        self.min_plane_gap = min(self.min_plane_gap, metrics["plane_gap_m"])
        self.max_speed = max(self.max_speed, metrics["flight_speed_m_s"])
        self.max_tilt = max(self.max_tilt, metrics["tilt_deg"])
        self.max_body_rate = max(self.max_body_rate, metrics["angular_velocity_error_rad_s"])
        for invalid, reason in (
            (np.any(state.position < self.position_min) or np.any(state.position > self.position_max), "FLIGHT_VOLUME_EXIT"),
            (metrics["flight_speed_m_s"] > self.safety["max_speed_m_s"], "EXCESSIVE_FLIGHT_SPEED"),
            (metrics["tilt_deg"] > self.safety["max_tilt_deg"], "EXCESSIVE_TILT"),
            (metrics["angular_velocity_error_rad_s"] > self.safety["max_body_rate_rad_s"], "EXCESSIVE_BODY_RATE"),
            (self.sat_run > self.safety["max_saturation_run_s"], "ACTUATOR_SATURATION"),
            (metrics["plane_gap_m"] < self.safety["min_gap_m"], "TERMINAL_OVERSHOOT"),
            (any(data.warning[i].number for i in range(len(data.warning))), "SIMULATOR_WARNING"),
        ):
            if invalid:
                self._fail(reason)
        terminal = self.terminal
        speed = terminal["normal_speed_m_s"]
        qualifies = (
            metrics["plane_gap_error_m"] <= terminal["gap_tolerance_m"]
            and metrics["tangent_position_error_m"] <= terminal["tangent_position_tolerance_m"]
            and abs(metrics["com_normal_speed_m_s"] - speed) <= terminal["normal_speed_tolerance_m_s"]
            and abs(metrics["material_normal_speed_m_s"] - speed) <= terminal["normal_speed_tolerance_m_s"]
            and metrics["com_tangent_speed_m_s"] <= terminal["tangent_speed_tolerance_m_s"]
            and metrics["material_tangent_speed_m_s"] <= terminal["tangent_speed_tolerance_m_s"]
            and metrics["attitude_error_deg"] <= terminal["attitude_tolerance_deg"]
            and metrics["angular_velocity_error_rad_s"] <= terminal["body_rate_tolerance_rad_s"])
        if not self.reasons and not self.arrived:
            # Closest joint approach while success was still possible.
            self.best_violation_ratio = min(self.best_violation_ratio, self.terminal_violation_ratio(metrics))
        # Safety wins at a shared timestamp. A valid arrival wins over expiry.
        if not self.reasons and qualifies and not self.arrived:
            self.arrival_time = timestamp - self.episode_start
            self.events.append(dict(time_s=timestamp, event="ARRIVAL_SUCCESS"))
        if not self.done and timestamp - self.episode_start >= self.horizon_s - 1e-12:
            self._fail("ARRIVAL_TIMEOUT")

    def integrate_interval(self, model, data, start_time, saturated=False):
        """Consume elapsed time and interval collisions before endpoint forward."""
        if self._final_result is not None:
            raise RuntimeError("Evaluator has a finalized result")
        start, end = float(start_time), float(data.time)
        if (self.snapshot_time is None
                or not np.isclose(start, self.snapshot_time, rtol=0, atol=1e-12)
                or not np.isclose(start, self.interval_end_time, rtol=0, atol=1e-12)):
            raise ValueError("Interval start must match the last consumed boundary")
        dt = end - start
        if not np.isclose(dt, model.opt.timestep, rtol=0, atol=1e-12) or dt <= 0:
            raise ValueError("Expected exactly one completed physics interval")
        self._external_contacts(model, data, sampling="COMPLETED_INTERVAL_SOLVER", interval_start=start)
        self.elapsed_time += dt
        self.saturated_time += dt if saturated else 0.
        self.sat_run = self.sat_run + dt if saturated else 0.
        self.max_sat_run = max(self.max_sat_run, self.sat_run)
        self.completed_intervals += 1
        self.interval_end_time = end

    def advance(self, model, data, before, saturated=False):
        """Use the same physical interval producer as the original contact task."""
        if self._final_result is not None or self.done:
            raise RuntimeError("Arrival evaluator has a terminal result")
        if (self.snapshot_time is None
                or not np.isclose(data.time, before.timestamp, rtol=0, atol=1e-12)
                or not np.isclose(before.timestamp, self.snapshot_time, rtol=0, atol=1e-12)
                or not np.isclose(before.timestamp, self.interval_end_time, rtol=0, atol=1e-12)):
            raise ValueError("Pre-arrival state must be the current interval start")
        state = advance_physics_interval(model, data, before,
                                         consume_interval=self.integrate_interval,
                                         saturated=saturated)
        self.snapshot(model, data, state)
        return state

    def irreversible_reasons(self, model, data):
        if self._final_result is not None:
            return list(self._final_result["failure_modes"])
        return sorted(self.reasons)

    def summary(self, model, data):
        if self._final_result is not None:
            return copy.deepcopy(self._final_result)
        metrics = {} if self.last_metrics is None else {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in self.last_metrics.items()}
        return dict(passed=bool(self.arrived and not self.reasons),
                    arrival_time_s=self.arrival_time,
                    failure_modes=sorted(self.reasons), events=copy.deepcopy(self.events),
                    termination_reason="ARRIVAL_SUCCESS" if self.arrived and not self.reasons else sorted(self.reasons),
                    episode_elapsed_s=float(data.time - self.episode_start), duration_s=float(data.time),
                    completed_physics_intervals=self.completed_intervals,
                    evaluation_protocol=self.protocol, terminal_metrics=metrics,
                    trajectory_length_m=self.path_length, min_bumper_plane_gap_m=float(self.min_plane_gap),
                    plane_gap_geometry="FACING_BUMPER_SPHERE_POINT_TO_ARMOR_FRONT_PLANE",
                    max_flight_speed_m_s=self.max_speed, max_tilt_deg=self.max_tilt,
                    max_body_rate_rad_s=self.max_body_rate,
                    saturation_fraction=self.saturated_time / self.elapsed_time if self.elapsed_time else 0.,
                    max_saturation_run_s=self.max_sat_run,
                    best_terminal_violation_ratio=(float(self.best_violation_ratio)
                                                   if np.isfinite(self.best_violation_ratio) else None),
                    task_id=self.task["task_id"], observation_schema=self.task["observation_schema"],
                    contact_recovery_success=False)

    def finalize(self, model, data):
        if self._final_result is None:
            if not self.done:
                raise RuntimeError("Cannot finalize an ongoing arrival episode")
            self._final_result = self.summary(model, data)
        return copy.deepcopy(self._final_result)
