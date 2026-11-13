"""Ideal terminal bottom-impact scenario, post-impact recovery evaluation and one trial.

The vehicle is placed a few millimetres in front of the armor plate in an ideal
terminal state (body z along the plate normal, bumper first) so that impact and
recovery are studied without any approach-flight policy.  Truth is read only here
and by the plant; the recovery controller receives a VehicleEstimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_metrics import armor_contacts
from .model import advance_physics_interval, target_frame, unit
from .imu import Bmi088Px4
from .plant import RateActuation
from .recovery import GRAVITY, ImpactDetector, RecoveryController, RecoveryGains

TARGET_GEOMS = ("armor", "rim_x_-1", "rim_x_1", "rim_y_-1", "rim_y_1")


@dataclass(frozen=True)
class ImpactCase:
    """Pre-impact state relative to the plate; angles in degrees, rates in body rad/s."""
    normal_speed_m_s: float = 2.25
    tangent_velocity_m_s: tuple = (0., 0.)
    tilt_deg: float = 0.
    tilt_azimuth_deg: float = 0.
    yaw_deg: float = 0.
    omega_body_rad_s: tuple = (0., 0., 0.)
    tangent_offset_m: tuple = (0., 0.)
    attitude_mode: str = "normal"  # "normal": body z along plate normal; "level": body z up
    gap_m: float = 0.02


@dataclass(frozen=True)
class RecoverySpec:
    mode: str = "stabilize"  # "none" holds the pre-impact command; "stabilize" recovers
    detection_threshold_g: float = 1.0
    detection_persistence: int = 1
    # "truth": physics specific force every step; "imu_filtered": PX4-filtered accel (IMU_ACCEL_CUTOFF);
    # "imu_raw": sensor-rate accel after full-scale clipping and the sensor's internal filter (needs imu_params).
    detection_source: str = "truth"
    takeover_delay_s: float = 0.
    setpoint_backoff_m: float = 0.3
    # Post-impact hover setpoint, relative to the position at takeover:
    #   "fixed":   setpoint_backoff_m along the plate normal;
    #   "patnaik": backoff_gain [s] * |pre-impact velocity| along the normal (Patnaik et al. 2021);
    #   "rebound": fixed first, then replaced once at detection + rebound_delay_s by
    #              position + backoff_gain [s] * velocity (translational part of Gaucher & Zhang's shift).
    setpoint_rule: str = "fixed"
    # Active retreat right after takeover: "none", "normal" (along the plate normal) or
    # "reverse" (opposite to the pre-impact velocity), for retreat_s at retreat_accel_m_s2.
    retreat: str = "none"
    retreat_s: float = 0.1
    retreat_accel_m_s2: float = 10.
    backoff_gain: float = 0.
    rebound_delay_s: float = 0.01
    horizon_after_contact_s: float = 2.0
    max_approach_s: float = 0.1
    inner_loop: str = "proxy"  # "proxy": plant CTBR proxy; "px4": PX4 rate PID + allocation (needs rate_params)
    gains: RecoveryGains = field(default_factory=RecoveryGains)


@dataclass(frozen=True)
class SuccessSpec:
    """Self-sustained hover: all bounds held continuously, off the plate, never on the ground."""
    max_body_rate_rad_s: float = 1.0
    max_tilt_deg: float = 10.
    max_vertical_speed_m_s: float = 0.3
    hold_s: float = 0.3
    stuck_contact_s: float = 0.2
    out_of_bounds_m: float = 2.0


def initial_state(plant, case):
    """Return (position, rotation, velocity, omega_body) for the ideal pre-impact case."""
    target = plant.physical["target"]
    plate = target_frame(target)
    n = unit(target["normal"])
    if case.attitude_mode == "normal":
        base = plate
    elif case.attitude_mode == "level":
        base = np.eye(3)
    else:
        raise ValueError("attitude_mode must be 'normal' or 'level'")
    base = base @ Rotation.from_euler("z", case.yaw_deg, degrees=True).as_matrix()
    azimuth = np.deg2rad(case.tilt_azimuth_deg)
    axis = np.cos(azimuth) * plate[:, 0] + np.sin(azimuth) * plate[:, 1]
    rotation = Rotation.from_rotvec(np.deg2rad(case.tilt_deg) * axis).as_matrix() @ base
    contact = plant.cfg["contact"]  # nominal bumper placement; drawn offsets stay hidden
    bumper = (np.asarray(target["center_m"], float) + plate[:, :2] @ np.asarray(case.tangent_offset_m, float)
              + (case.gap_m + contact["radius_m"]) * n)
    position = bumper - rotation @ np.asarray(contact["offset_body_m"], float)
    velocity = -case.normal_speed_m_s * n + plate[:, :2] @ np.asarray(case.tangent_velocity_m_s, float)
    return position, rotation, velocity, np.asarray(case.omega_body_rad_s, float)


def specific_force_body(plant):
    """Accelerometer truth at the body origin (COM): R^T (a - g)."""
    rotation = plant.data.xmat[plant.model.body("uav").id].reshape(3, 3)
    return rotation.T @ (plant.data.qacc[:3] + [0., 0., GRAVITY])


class RecoveryEvaluator:
    """Per-physics-step bookkeeping; classifies one post-impact episode."""

    def __init__(self, plant, success=SuccessSpec()):
        m = plant.model
        self.plant, self.success = plant, success
        self.target_ids = {m.geom(name).id for name in TARGET_GEOMS}
        self.ground_id = m.geom("ground").id
        self.armor_id = m.geom("armor").id
        self.normal = unit(plant.physical["target"]["normal"])
        # Geoms that count as the intended bottom contact (name substrings), e.g. the bumper or feet.
        self.bottom_names = tuple(plant.cfg["contact"].get("bottom_geoms", ["bumper"]))
        self.first_contact = None
        self.contact_time = 0.
        self.contact_episodes = 0
        self.in_contact = False
        self.touched = set()
        self.ground = False
        self.peak_specific_force_g = 0.
        self.peak_normal_force_N = 0.
        self.max_tilt_deg = 0.
        self.min_z = np.inf
        self.max_distance = 0.
        self.hold_start = None
        self.recovered_at = None
        self.impact_position = None
        self.final_tilt_deg = None
        self.separated_at = None
        self.secondary_impact = False

    def consume_interval(self, timestamp, dt, force):
        """Consume the solve that actually advanced velocity, before mj_forward.

        The observation is delivered at the end of the completed interval. Pose
        and recovery bounds are evaluated separately on the refreshed endpoint.
        """
        m, d = self.plant.model, self.plant.data
        touching = False
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if self.ground_id in (g1, g2):
                self.ground = True
            hit = self.target_ids & {g1, g2}
            if hit:
                touching = True
                other = g2 if g1 in hit else g1
                self.touched.add(m.geom(other).name)
        if touching:
            for record in armor_contacts(m, d, self.armor_id, self.normal):
                self.peak_normal_force_N = max(self.peak_normal_force_N, record["compressive_normal_force_N"])
            if self.first_contact is None:
                self.first_contact = timestamp
                self.impact_position = d.qpos[:3].copy()
            if not self.in_contact:
                self.contact_episodes += 1
                # A re-contact after >= 20 ms clear of the plate is a secondary impact, not chatter.
                if self.separated_at is not None and timestamp - self.separated_at >= 0.02 - 1e-9:
                    self.secondary_impact = True
            self.contact_time += dt
        if self.in_contact and not touching:
            self.separated_at = timestamp
        self.in_contact = touching
        if self.first_contact is not None:
            self.peak_specific_force_g = max(self.peak_specific_force_g,
                                             float(np.linalg.norm(force)) / GRAVITY)

    def update_state(self, state):
        """Evaluate recovery using endpoint pose and the consumed interval contact."""
        if self.first_contact is None:
            return
        tilt = float(np.degrees(np.arccos(np.clip(state.rotation[2, 2], -1., 1.))))
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        self.final_tilt_deg = tilt
        self.min_z = min(self.min_z, float(state.position[2]))
        self.max_distance = max(self.max_distance, float(np.linalg.norm(state.position - self.impact_position)))
        s = self.success
        steady = (not self.in_contact and np.linalg.norm(state.omega_body) < s.max_body_rate_rad_s
                  and tilt < s.max_tilt_deg and abs(state.velocity[2]) < s.max_vertical_speed_m_s)
        if self.recovered_at is None:
            if steady:
                self.hold_start = state.timestamp if self.hold_start is None else self.hold_start
                if state.timestamp - self.hold_start >= s.hold_s - 1e-9:
                    self.recovered_at = state.timestamp
            else:
                self.hold_start = None

    @property
    def terminal(self):
        return self.ground or self.max_distance > self.success.out_of_bounds_m

    def outcome(self):
        if self.first_contact is None:
            return "NO_CONTACT"
        if self.ground:
            return "GROUND"
        if self.max_distance > self.success.out_of_bounds_m:
            return "OUT_OF_BOUNDS"
        if self.recovered_at is not None:
            return "RECOVERED"
        if self.contact_time > self.success.stuck_contact_s:
            # Resting inverted on the plate is the tripped-and-flipped mode, not a pinned hover.
            return "FLIPPED_ON_PLATE" if self.in_contact and self.final_tilt_deg > 90. else "STUCK_ON_PLATE"
        if self.max_tilt_deg > 90.:
            return "FLIPPED"
        return "NOT_SETTLED"


def run_trial(plant, case, spec=RecoverySpec(), success=SuccessSpec(), *, rate_params=None, imu_params=None,
              imu_seed=None, trace=False):
    """Fly one ideal impact on an already-built plant; return a JSON-ready summary."""
    if spec.retreat not in ("none", "normal", "reverse"):
        raise ValueError("retreat must be 'none', 'normal' or 'reverse'")
    if spec.setpoint_rule not in ("fixed", "patnaik", "rebound"):
        raise ValueError("setpoint_rule must be 'fixed', 'patnaik' or 'rebound'")
    if spec.detection_source not in ("truth", "imu_filtered", "imu_raw"):
        raise ValueError("detection_source must be 'truth', 'imu_filtered' or 'imu_raw'")
    if spec.detection_source != "truth" and imu_params is None:
        raise ValueError("IMU detection needs imu_params")
    if spec.inner_loop not in ("proxy", "px4"):
        raise ValueError("inner_loop must be 'proxy' or 'px4'")
    if spec.inner_loop == "px4" and rate_params is None:
        raise ValueError("The PX4 inner loop needs rate_params")
    position, rotation, velocity, omega = initial_state(plant, case)
    state = plant.reset(position=position, velocity=velocity, rotation=rotation, omega_body=omega)
    m, d = plant.model, plant.data
    mass, hover = float(plant.info["mass_kg"]), plant.hover_force
    n = unit(plant.physical["target"]["normal"])
    physics_dt = float(m.opt.timestep)
    control_steps = max(1, int(round(plant.cfg["control_dt_s"] / physics_dt)))
    detector = ImpactDetector(spec.detection_threshold_g, spec.detection_persistence)
    imu = Bmi088Px4(imu_params, imu_seed) if imu_params is not None else None
    peak_raw_g = peak_filtered_g = 0.
    evaluator = RecoveryEvaluator(plant, success)
    controller = None
    command = np.array([hover, 0., 0., 0.])
    saturated_time = 0.
    approach_velocity = state.velocity.copy()
    rebound_applied = False
    actuation = RateActuation(plant, spec.inner_loop, rate_params)
    rows = []
    start = state.timestamp
    step = 0
    thrust_estimate = hover

    def consume_interval(model, data, before_time, was_saturated):
        nonlocal peak_raw_g, peak_filtered_g
        force = specific_force_body(plant)
        timestamp = float(data.time)
        evaluator.consume_interval(timestamp, physics_dt, force)
        if imu is not None:
            imu.update(timestamp, force, data.qvel[3:6])
            if imu.accel.new_sample and evaluator.first_contact is not None:
                peak_raw_g = max(peak_raw_g, float(np.linalg.norm(imu.accel.raw)) / GRAVITY)
                peak_filtered_g = max(peak_filtered_g, float(np.linalg.norm(imu.accel.filtered)) / GRAVITY)
        if spec.detection_source == "truth":
            detector.update(timestamp, force, thrust_estimate, mass)
        elif imu.accel.new_sample:
            signal = imu.accel.raw if spec.detection_source == "imu_raw" else imu.accel.filtered
            detector.update(timestamp, signal, thrust_estimate, mass)

    while True:
        if step % control_steps == 0:
            if (spec.mode == "stabilize" and controller is None and detector.detected_at is not None
                    and state.timestamp - detector.detected_at >= spec.takeover_delay_s - 1e-9):
                backoff = (spec.backoff_gain * float(np.linalg.norm(approach_velocity))
                           if spec.setpoint_rule == "patnaik" else spec.setpoint_backoff_m)
                direction = (None if spec.retreat == "none" else n if spec.retreat == "normal"
                             else -approach_velocity / max(np.linalg.norm(approach_velocity), 1e-9))
                setpoint = state.position + backoff * (n if direction is None else direction)
                controller = RecoveryController(mass, 4 * plant.maximum, setpoint, spec.gains,
                                                retreat_direction=direction,
                                                retreat_until=state.timestamp + spec.retreat_s,
                                                retreat_accel_m_s2=spec.retreat_accel_m_s2)
            if (controller is not None and spec.setpoint_rule == "rebound" and not rebound_applied
                    and state.timestamp - detector.detected_at >= spec.rebound_delay_s - 1e-9):
                controller.setpoint = state.position + spec.backoff_gain * state.velocity
                rebound_applied = True
            if controller is not None:
                command = controller.command(state)
        saturated = actuation.apply(command, update_proxy=step % control_steps == 0)
        # Actuator state used by the completed solve, not its newly integrated value.
        thrust_estimate = float(np.sum(d.act))

        state = advance_physics_interval(m, d, state, consume_interval=consume_interval, saturated=saturated)
        step += 1
        saturated_time += physics_dt * saturated
        evaluator.update_state(state)
        if evaluator.first_contact is None:
            approach_velocity = state.velocity.copy()
        if trace:
            rows.append(np.r_[state.timestamp - start, state.position, state.velocity,
                              Rotation.from_matrix(state.rotation).as_rotvec(), state.omega_body,
                              command, float(evaluator.in_contact)])
        if evaluator.first_contact is None:
            if state.timestamp - start > spec.max_approach_s:
                break
        elif evaluator.terminal or state.timestamp - evaluator.first_contact >= spec.horizon_after_contact_s - 1e-9:
            break
    tc = evaluator.first_contact
    result = dict(
        outcome=evaluator.outcome(),
        recovery_time_s=None if evaluator.recovered_at is None or tc is None else evaluator.recovered_at - tc,
        detection_delay_s=None if detector.detected_at is None or tc is None else detector.detected_at - tc,
        detected=detector.detected_at is not None,
        false_alarm=detector.detected_at is not None and (tc is None or detector.detected_at < tc - 1e-9),
        imu_peak_raw_g=None if imu is None else peak_raw_g,
        imu_peak_filtered_g=None if imu is None else peak_filtered_g,
        imu_accel_clipped_samples=None if imu is None else imu.accel.clipped_samples,
        contact_time_s=evaluator.contact_time,
        contact_episodes=evaluator.contact_episodes,
        secondary_impact=evaluator.secondary_impact,
        final_tilt_deg=evaluator.final_tilt_deg,
        touched_geoms=sorted(evaluator.touched),
        bottom_only=all(any(b in g for b in evaluator.bottom_names) for g in evaluator.touched),
        peak_specific_force_g=evaluator.peak_specific_force_g,
        peak_normal_force_N=evaluator.peak_normal_force_N,
        max_tilt_deg=evaluator.max_tilt_deg,
        altitude_loss_m=None if tc is None else float(evaluator.impact_position[2] - evaluator.min_z),
        max_distance_m=evaluator.max_distance,
        final_drift_m=None if controller is None else float(np.linalg.norm(state.position - controller.setpoint)),
        setpoint_offset_m=None if controller is None or tc is None else (
            controller.setpoint - evaluator.impact_position).tolist(),
        approach_speed_m_s=float(np.linalg.norm(approach_velocity)),
        motor_saturation_time_s=saturated_time,
        actuator_capability=plant.actuator_capability(actuation.per_output),
    )
    if trace:
        result["trace_columns"] = ["t", "px", "py", "pz", "vx", "vy", "vz", "rx", "ry", "rz",
                                   "wx", "wy", "wz", "T", "p_cmd", "q_cmd", "r_cmd", "contact"]
        result["trace"] = np.asarray(rows)
    return result
