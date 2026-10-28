"""Continuous local Ver2 proxy mission; no resets or pose writes between phases.

The shared physics package owns the plant, trajectory tracking, rate actuation,
impact detector and recovery controller. This module owns mission sequencing and
experiment evidence. All feedback is explicitly MuJoCo truth, not firmware SITL.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .impact_recovery import ROOT, load_study, provenance
from rm27_contact_physics.contact_metrics import armor_contacts, point_approach_speed
from rm27_contact_physics.control import Septic, Skill
from rm27_contact_physics.impact import RecoveryEvaluator, specific_force_body, TARGET_GEOMS
from rm27_contact_physics.model import TargetState, advance_physics_interval, target_frame, unit
from rm27_contact_physics.plant import RM27Plant, RateActuation
from rm27_contact_physics.recovery import ImpactDetector, RecoveryController
from rm27_contact_physics.teacher_interface import project


PHASES = ("REST", "TAKEOFF", "TRANSIT", "APPROACH", "RECOVER", "RETURN", "LAND", "DISARMED", "COMPLETE")
TRACE_COLUMNS = ["time_s", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz",
                 "wx", "wy", "wz", *[f"motor_state_{i}" for i in range(4)],
                 *[f"motor_command_{i}" for i in range(4)], "phase", "target_contact", "pad_contact"]


def make_plant(study, config, contact, seed):
    base = copy.deepcopy(study["base"])
    base["landing_pad"] = config["landing_pad"]
    base["controller"].update(config.get("tracking", {}))
    values = config["contacts"][contact]
    base["target"].update(values)
    base["uncertainty"].update(friction_range=[values["friction"]]*2,
                               contact_damping_range=[values["solref_damping_ratio"]]*2,
                               contact_time_scale_range=[1., 1.])
    return RM27Plant(base, study["sources_plant"], physical_seed=seed,
                     physical_scale=study["cfg"]["physical_scale"])


def simulate(study, config, contact="soft", seed=None):
    """Return physical mission evidence and a sampled state/actuator trajectory."""
    plant = make_plant(study, config, contact, seed)
    m, d, p = plant.model, plant.data, config["mission"]
    dt = float(m.opt.timestep)
    divider = int(round(plant.cfg["control_dt_s"] / dt))
    pad = config["landing_pad"]
    foot_height = plant.vehicle_config["com_height_above_bottom_m_assumed"]
    home = np.r_[pad["center_xy_m"], pad["height_m"]+foot_height]
    hover = np.r_[home[:2], p["takeoff_height_m"]]
    # Exactly one initial reset. The drawn dynamics/bumper remain hidden from
    # the nominal controllers; target pose is explicit truth guidance here.
    state = plant.reset(position=home+[0., 0., .001], velocity=np.zeros(3),
                        rotation=np.eye(3), omega_body=np.zeros(3))
    d.act[:] = 0.
    d.ctrl[:] = 0.
    mujoco.mj_forward(m, d)
    target = TargetState(np.asarray(plant.physical["target"]["center_m"]),
                         unit(plant.physical["target"]["normal"]), np.zeros((3, 3)), 0.)
    parameters = dict(study["base"]["nominal_skill"], duration_s=p["approach_duration_s"],
                      normal_speed_m_s=p["approach_normal_speed_m_s"],
                      terminal_specific_thrust_m_s2=p["terminal_specific_thrust_m_s2"],
                      penetration_m=p["penetration_m"])
    tracker = Skill(plant.cfg, parameters, state, target, plant.info, marker_config=plant.marker_config)
    tracker.motor_estimate[:] = 0.
    tracker.previous_motor_command[:] = 0.
    actuation = RateActuation(plant, "proxy")
    evaluator = RecoveryEvaluator(plant, study["success"])
    detector = ImpactDetector()
    stage = target.position + p["staging_distance_m"]*target.normal
    stage -= np.asarray(plant.cfg["contact"]["offset_body_m"])
    zeros = np.zeros(3)
    phase, entered, stable_since = "REST", 0., None
    events = [dict(phase=phase, time_s=0., position_m=state.position.tolist())]
    frames, trajectory, destination, recovery = [], None, home.copy(), None
    descent_start, descent_deadline = None, None
    failure, impact, landing, recovery_result = None, None, None, None
    target_touch = pad_touch = ground_touch = False
    sat_time = 0.
    max_tilt = max_rate = max_speed = 0.
    saturation_by_phase = dict.fromkeys(PHASES, 0.)
    command = np.zeros(4)

    def transition(name, goal=None, duration=None):
        nonlocal phase, entered, stable_since, trajectory, destination, tracker, recovery
        nonlocal descent_start, descent_deadline
        phase, entered, stable_since = name, state.timestamp, None
        events.append(dict(phase=name, time_s=entered, position_m=state.position.tolist()))
        if name == "APPROACH":
            current_target = dataclasses.replace(target, timestamp=state.timestamp)
            # Retain the command-driven observer and inferred thrust/mass ratio
            # learned in free flight when constructing the terminal trajectory.
            tracker.trajectory = Skill(plant.cfg, parameters, state, current_target, plant.info,
                                       marker_config=plant.marker_config).trajectory
        elif name == "RECOVER":
            recovery = RecoveryController(plant.info["mass_kg"], 4*plant.maximum,
                                          state.position+p["backoff_m"]*target.normal)
        elif name == "LAND":
            descent_start = state.position.copy()
            descent_deadline = entered+(descent_start[2]-home[2])/p["descent_speed_m_s"]+p["settle_timeout_s"]
        elif goal is not None:
            destination = np.asarray(goal, float)
            trajectory = Septic((state.position, state.velocity, zeros, zeros),
                                 (destination, zeros, zeros, zeros), duration)

    def held(condition, duration):
        nonlocal stable_since
        stable_since = (state.timestamp if stable_since is None else stable_since) if condition else None
        return stable_since is not None and state.timestamp-stable_since >= duration-1e-9

    def settled():
        return (np.linalg.norm(state.position-destination) < p["waypoint_position_tolerance_m"]
                and np.linalg.norm(state.velocity) < p["waypoint_speed_tolerance_m_s"]
                and np.linalg.norm(state.omega_body) < study["success"].max_body_rate_rad_s)

    def consume_interval(model, data, before_time, saturated):
        nonlocal target_touch, pad_touch, ground_touch, impact
        pairs = [(model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name)
                 for c in data.contact[:data.ncon]]
        target_touch = any(set(pair) & set(TARGET_GEOMS) for pair in pairs)
        pad_touch = any(any(name.startswith("landing_pad_") for name in pair) for pair in pairs)
        ground_touch = any("ground" in pair for pair in pairs)
        if phase in ("APPROACH", "RECOVER"):
            force = specific_force_body(plant)
            evaluator.consume_interval(float(data.time), dt, force)
            detector.update(float(data.time), force, thrust_estimate, plant.info["mass_kg"])
            if target_touch and impact is None:
                records = armor_contacts(model, data, model.geom("armor").id, target.normal)
                intended = [r for r in records if "bumper" in r["geoms"]]
                record = intended[0] if intended else (records[0] if records else None)
                position = np.array(record["position_world_m"]) if record else state.position
                velocity = state.velocity + np.cross(state.rotation @ state.omega_body, position-state.position)
                normal_speed = point_approach_speed(state.velocity, state.rotation @ state.omega_body,
                                                    position-state.position, zeros, target.normal)
                tangent_speed = float(np.linalg.norm(velocity+(normal_speed*target.normal)))
                attitude_error = float(np.degrees(np.arccos(np.clip(state.rotation[:, 2] @ target.normal, -1., 1.))))
                local_point = target_frame(plant.physical["target"]).T @ (position-target.position)
                inside = np.all(np.abs(local_point[:2]) < np.array(plant.physical["target"]["dimensions_m"][:2])/2-p["contact_edge_margin_m"])
                valid = (bool(intended) and bool(inside) and p["contact_normal_speed_range_m_s"][0] <= normal_speed
                         <= p["contact_normal_speed_range_m_s"][1] and tangent_speed <= p["contact_tangent_speed_max_m_s"]
                         and attitude_error <= p["contact_attitude_error_max_deg"])
                impact = dict(time_s=float(data.time), valid=valid, normal_speed_m_s=normal_speed,
                              tangent_speed_m_s=tangent_speed, attitude_error_deg=attitude_error,
                              inside_armor_margin=bool(inside), bottom_contact=bool(intended),
                              position_m=position.tolist(), contacts=pairs)

    for step in range(int(round(p["max_duration_s"]/dt))):
        elapsed = state.timestamp-entered
        if step % divider == 0:
            if phase == "REST" and elapsed >= p["rest_s"]:
                transition("TAKEOFF", hover, p["takeoff_duration_s"])
            elif phase in ("TAKEOFF", "TRANSIT", "RETURN"):
                if elapsed >= trajectory.duration and held(settled(), p["waypoint_hold_s"]):
                    if phase == "TAKEOFF":
                        transition("TRANSIT", stage, p["transit_duration_s"])
                    elif phase == "TRANSIT":
                        transition("APPROACH")
                    else:
                        transition("LAND")
                elif elapsed > trajectory.duration+p["settle_timeout_s"]:
                    failure = phase+"_TIMEOUT"
            elif phase == "APPROACH":
                if detector.detected_at is not None:
                    if impact is None:
                        failure = "FALSE_IMPACT_DETECTION"
                    elif not impact["valid"]:
                        failure = "INVALID_IMPACT"
                    else:
                        transition("RECOVER")
                elif elapsed > p["approach_duration_s"]+p["miss_timeout_s"]:
                    failure = "NO_IMPACT_DETECTION"
            elif phase == "RECOVER":
                if (evaluator.recovered_at is not None and not evaluator.terminal
                        and evaluator.recovered_at-evaluator.first_contact <= p["recovery_deadline_s"]+1e-9):
                    recovery_result = dict(outcome=evaluator.outcome(),
                        recovery_time_s=evaluator.recovered_at-evaluator.first_contact,
                        altitude_loss_m=float(evaluator.impact_position[2]-evaluator.min_z),
                        contact_time_s=evaluator.contact_time, secondary_impact=evaluator.secondary_impact,
                        touched_geoms=sorted(evaluator.touched), setpoint_m=recovery.setpoint.tolist())
                    # Preserve the controller state at handoff; the new reference
                    # begins at the current physical position and velocity.
                    tracker.last_time = None
                    # Reinitialize the command-driven observer from the last
                    # issued commands, never hidden rotor state/effectiveness.
                    tracker.motor_estimate = d.ctrl.copy()
                    tracker.previous_motor_command = d.ctrl.copy()
                    tracker.previous_velocity = state.velocity.copy()
                    tracker.previous_rotation = state.rotation.copy()
                    transition("RETURN", hover, p["return_duration_s"])
                elif state.timestamp-evaluator.first_contact > p["recovery_deadline_s"]:
                    failure = "RECOVERY_"+evaluator.outcome()
            elif phase == "LAND":
                tilt = float(np.degrees(np.arccos(np.clip(state.rotation[2, 2], -1., 1.))))
                contact_ok = (pad_touch and np.linalg.norm(state.position[:2]-home[:2]) < p["landing_xy_tolerance_m"]
                              and abs(state.position[2]-home[2]) < p["landing_height_tolerance_m"]
                              and np.linalg.norm(state.velocity) < p["landing_speed_max_m_s"]
                              and tilt < p["landing_tilt_max_deg"])
                if held(contact_ok, p["landing_hold_s"]):
                    landing = dict(time_s=state.timestamp, position_m=state.position.tolist(),
                                   speed_m_s=float(np.linalg.norm(state.velocity)), tilt_deg=tilt)
                    transition("DISARMED")
                elif state.timestamp > descent_deadline:
                    failure = "LANDING_TIMEOUT"
            elif phase == "DISARMED":
                resting = (pad_touch and np.linalg.norm(state.position[:2]-home[:2]) < p["landing_xy_tolerance_m"]
                           and abs(state.position[2]-home[2]) < p["landing_height_tolerance_m"]
                           and state.rotation[2, 2] > np.cos(np.deg2rad(p["landing_tilt_max_deg"]))
                           and np.linalg.norm(state.velocity) < p["resting_speed_max_m_s"]
                           and np.linalg.norm(state.omega_body) < p["resting_rate_max_rad_s"]
                           and np.max(np.abs(d.actuator_force)) < p["stopped_motor_force_max_N"])
                if held(resting, p["disarmed_hold_s"]):
                    transition("COMPLETE")
                elif elapsed > p["settle_timeout_s"]:
                    failure = "DISARMED_REST_TIMEOUT"

            if phase in ("REST", "DISARMED", "COMPLETE"):
                command = np.zeros(4)
                saturated = False
                d.ctrl[:] = 0.
            elif phase == "RECOVER":
                command = recovery.command(state)
                saturated = actuation.apply(command)
            else:
                elapsed = state.timestamp-entered
                if phase == "APPROACH":
                    desired = tracker.trajectory.sample(elapsed)
                    if elapsed > tracker.trajectory.duration:
                        h = elapsed-tracker.trajectory.duration
                        pos, vel, acc, jerk = desired
                        desired = (pos+vel*h+.5*acc*h*h, vel+acc*h, acc, jerk)
                elif phase == "LAND":
                    # Continue the commanded descent until measured support
                    # contact. A fixed terminal height can hover above the pad
                    # under thrust/mass bias, and cannot serve as a touchdown.
                    position = np.r_[home[:2], descent_start[2]-p["descent_speed_m_s"]*elapsed]
                    desired = (position, np.array([0., 0., -p["descent_speed_m_s"]]), zeros, zeros)
                else:
                    desired = trajectory.sample(elapsed)
                # Support reactions must not be learned as rotor effectiveness.
                motors, saturated = tracker.track(state, desired,
                    specific_thrust_ratio=tracker.specific_thrust_ratio if pad_touch else None)
                action = project(motors, state.omega_body, plant.inertia, plant.mixer, plant.rate_gain)
                command = np.r_[action.collective_force_N, action.body_rates_flu_rad_s]
                saturated = actuation.apply(command) or saturated
        if failure or phase == "COMPLETE":
            break
        thrust_estimate = float(np.sum(d.act))
        state = advance_physics_interval(m, d, state, consume_interval=consume_interval, saturated=saturated)
        sat_time += dt*saturated
        saturation_by_phase[phase] += dt*saturated
        tilt = float(np.degrees(np.arccos(np.clip(state.rotation[2, 2], -1., 1.))))
        max_tilt = max(max_tilt, tilt)
        max_rate = max(max_rate, float(np.linalg.norm(state.omega_body)))
        max_speed = max(max_speed, float(np.linalg.norm(state.velocity)))
        if phase in ("APPROACH", "RECOVER"):
            evaluator.update_state(state)
        if not np.isfinite(np.r_[d.qpos, d.qvel, d.act]).all():
            failure = "NONFINITE"
        elif ground_touch:
            failure = "GROUND_CONTACT"
        elif target_touch and phase not in ("APPROACH", "RECOVER"):
            failure = "UNEXPECTED_TARGET_CONTACT"
        elif pad_touch and phase not in ("REST", "TAKEOFF", "LAND", "DISARMED"):
            failure = "UNEXPECTED_PAD_CONTACT"
        elif max_rate > p["max_body_rate_rad_s"] or state.position[2] > p["max_height_m"] or np.linalg.norm(state.position-home) > p["max_distance_from_home_m"]:
            failure = "FLIGHT_ENVELOPE"
        if step % p["trace_stride"] == 0 or failure:
            frames.append(np.r_[state.timestamp, d.qpos, d.qvel, d.act, d.ctrl,
                                PHASES.index(phase), target_touch, pad_touch])
        if failure:
            break
    if phase != "COMPLETE" and failure is None:
        failure = "MISSION_TIMEOUT"
    frames.append(np.r_[state.timestamp, d.qpos, d.qvel, d.act, d.ctrl,
                        PHASES.index(phase), target_touch, pad_touch])
    result = dict(passed=phase == "COMPLETE" and failure is None, failure=failure, phase=phase,
                  contact=contact, physical_seed=seed, physical_draw=plant.physical["draw"],
                  initial_state_policy="fixed upright at pad; scenario start_dp/dv/rotvec are not applied",
                  events=events, impact=impact, recovery=recovery_result, landing=landing,
                  detection_time_s=detector.detected_at,
                  inferred_specific_thrust_ratio=tracker.specific_thrust_ratio,
                  duration_s=state.timestamp, motor_saturation_time_s=sat_time,
                  saturation_by_phase_s=saturation_by_phase, max_tilt_deg=max_tilt,
                  max_body_rate_rad_s=max_rate, max_speed_m_s=max_speed,
                  final_position_m=state.position.tolist(), final_velocity_m_s=state.velocity.tolist(),
                  final_motor_force_N=d.actuator_force.tolist(), final_motor_command=d.ctrl.tolist(),
                  actuator_capability=plant.actuator_capability(), reset_count=1)
    return result, np.asarray(frames), plant


def _run_one(study, config, output, task):
    contact, group, repeat, seed = task
    result, trace, plant = simulate(study, config, contact, seed)
    name = f"{contact}_{group}_{repeat:03d}"
    np.savez_compressed(output/(name+".npz"), trace=trace, columns=TRACE_COLUMNS, phases=PHASES)
    if group == "nominal" and repeat == 0:
        mujoco.mj_saveModel(plant.model, str(output/(name+".mjb")))
    return dict(group=group, repeat=repeat, trajectory=name+".npz", **result)


def run(config_path, output, *, workers=1):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("Expected schema version 1 and positive worker count")
    study = load_study(ROOT/config["base_study"])
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config["seed"])
    seeds = rng.choice(2**31, size=config["varied_trials"], replace=False).tolist()
    tasks = [(contact, group, i, seed) for contact in config["contacts"]
             for group, group_seeds in (("nominal", [None]*config["nominal_trials"]), ("varied", seeds))
             for i, seed in enumerate(group_seeds)]
    if not tasks:
        raise ValueError("Mission batch needs at least one trial")
    sources = dict(study["sources"], base_study=study["path"])
    metadata = dict(config=config, base_config=study["cfg"], base_experiment=study["base"],
                    success_definition=dataclasses.asdict(study["success"]),
                    input_snapshots={k: json.loads(v.read_text()) for k, v in sources.items() if v.suffix == ".json"},
                    provenance=provenance(config_path, dict(study["cfg"], seed=config["seed"]), sources),
                    scope=config["scope"], trace_columns=TRACE_COLUMNS, phases=PHASES)
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    execute = partial(_run_one, study, config, output)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = []
    try:
        batches = pool.map(execute, tasks) if pool else map(execute, tasks)
        with (output/"trials.jsonl").open("w") as handle:
            for row in batches:
                results.append(row)
                handle.write(json.dumps(row)+"\n")
                handle.flush()
                if len(results) % 10 == 0 or len(results) == len(tasks):
                    print(f'Missions {len(results)}/{len(tasks)}; successful {sum(r["passed"] for r in results)}', flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    groups = {}
    for contact in config["contacts"]:
        for group in ("nominal", "varied"):
            rows = [r for r in results if r["contact"] == contact and r["group"] == group]
            if not rows:
                continue
            k, n = sum(r["passed"] for r in rows), len(rows)
            failures = {}
            for row in rows:
                if row["failure"]:
                    failures[row["failure"]] = failures.get(row["failure"], 0)+1
            z = 1.96
            center = (k+z*z/2)/(n+z*z)
            half = z*np.sqrt(k*(n-k)/n+z*z/4)/(n+z*z)
            groups[f"{contact}/{group}"] = dict(trials=n, succeeded=k, success_rate=k/n,
                wilson95=None if group == "nominal" else [center-half, center+half],
                accepted=k == n if group == "nominal" else k >= config["minimum_varied_successes"],
                failures=failures,
                duration_s=dict(zip(("p50", "p95", "max"), np.quantile([r["duration_s"] for r in rows], [.5, .95, 1.]).tolist())))
            metrics = {"motor_saturation_time_s": [r["motor_saturation_time_s"] for r in rows],
                       "recovery_time_s_among_recovered": [r["recovery"]["recovery_time_s"] for r in rows if r["recovery"]],
                       "altitude_loss_m_among_recovered": [r["recovery"]["altitude_loss_m"] for r in rows if r["recovery"]],
                       "impact_normal_speed_m_s": [r["impact"]["normal_speed_m_s"] for r in rows if r["impact"]]}
            groups[f"{contact}/{group}"]["metrics"] = {
                name: dict(samples=len(values), **dict(zip(("p50", "p95", "max"), np.quantile(values, [.5, .95, 1.]).tolist())))
                for name, values in metrics.items() if values}
    report = dict(completed=len(results) == len(tasks), passed=all(g["accepted"] for g in groups.values()),
                  trials=len(results), groups=groups, scope=config["scope"], seed=config["seed"],
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
