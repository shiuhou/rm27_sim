"""Simulation-only single-vehicle takeoff, hover, return and dock landing gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import select
import socket
import subprocess
import time
import numpy as np
from pymavlink import mavutil
from ..simulation.micro_plant import MicroPlant, ROOT, BRIDGE_ROOT, to_ap
from .bridge.bridge import Bridge
from .bridge.launch import EXPECTED_PARAMS


def read_params(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, value = line.split()
            result[name] = float(value)
    return result


def run(scene, output, instance=31, *, view=False, speedup=None, dynamics=None, route=None):
    speedup = (1.0 if view else 5.0) if speedup is None else speedup
    if not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("speedup must be finite and greater than zero")
    scene, output = Path(scene).resolve(), Path(output).resolve()
    plant = MicroPlant(scene, dynamics_profile=dynamics)
    bridge = Bridge(plant)
    if route is not None and np.linalg.norm(route.points[0]-(plant.origin+[0, 0, .6])) > .001:
        raise ValueError('Route must start at the tested dock hover position')
    physics_deadline = 160 if route is None else route.physics_deadline
    if plant.data.ncon:
        raise ValueError("Initial scene contact requires review")
    from rm27_sim.paths import resolve_asset
    ap = resolve_asset("rm27/autopilot/vendor/ardupilot")
    binary = ap / "build/sitl/bin/arducopter"
    micro_params = ROOT / "configs/micro_hover.parm"
    expected = dict(EXPECTED_PARAMS, **read_params(micro_params))
    defaults = [
        ap / "Tools/autotest/default_params/copter.parm",
        ap / "Tools/autotest/default_params/copter-X.parm",
        BRIDGE_ROOT / "hover.parm",
        micro_params,
    ]
    command = [
        str(binary),
        "--model",
        "JSON:127.0.0.1",
        "--rate",
        "500",
        "--speedup",
        f"{speedup:g}",
        "--instance",
        str(instance),
        "--defaults",
        ",".join(map(str, defaults)),
        "--home",
        "-35.363261,149.165230,584,0",
        "--wipe",
    ]
    output.mkdir(parents=True, exist_ok=False)
    if dynamics is not None:
        # Snapshot provenance so replay does not depend on an edited profile path.
        import copy
        from ..experiments.calibration import load_profile
        profile = copy.deepcopy(load_profile(dynamics))
        for evidence in profile['evidence']:
            evidence['path'] = str((Path(dynamics).resolve().parent / evidence['path']).resolve())
        (output / 'dynamics.json').write_text(json.dumps(profile, indent=2))
    metadata = {
        "command": command,
        "scene": str(scene),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "parameter_files": {str(p): p.read_text() for p in defaults},
        "scope": "single-vehicle dock takeoff/hover/return/land/disarm; no target mission",
        "guidance": "ArduPilot GUIDED, simulation-truth position and velocity correction for precision return",
        "origin_nwu": plant.origin.tolist(),
        "vehicle": "drone_1",
        "ardupilot_source_commit": subprocess.check_output(
            ["git", "-C", str(ap), "rev-parse", "HEAD"], text=True
        ).strip(),
        "mass_kg": float(plant.model.body_mass[plant.body]),
        "inertia_kg_m2": plant.model.body_inertia[plant.body].tolist(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "live_view": view,
        "requested_speedup": speedup,
    }
    if dynamics is not None:
        metadata['dynamics_profile'] = 'dynamics.json'
        metadata['dynamics_profile_sha256'] = hashlib.sha256((output/'dynamics.json').read_bytes()).hexdigest()
    if route is not None:
        metadata['scope'] = 'Red-dock single-vehicle blue-armor contact and return; simulation truth assisted'
        metadata['route_audit'] = route.audit
        metadata['mission'] = route.mission
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    summary = {
        "passed": False,
        "failure": None,
        "events": [],
        "parameters_verified": False,
    }
    telemetry = {"mode": None, "armed": False, "ready": False, "ekf_flags": 0}
    params = {}
    phase = "INITIALIZE"
    phase_start = 0.0
    stable_since = None
    next_action, next_send, next_heartbeat = 15.0, 0.0, 0.0
    requested = False
    initial_yaw = math.atan2(
        plant.data.xmat[plant.body].reshape(3, 3)[1, 0],
        plant.data.xmat[plant.body].reshape(3, 3)[0, 0],
    )
    max_tilt, max_force, max_penetration, max_inactive_drift = 0.0, 0.0, 0.0, 0.0
    hover_samples = []
    landing = None
    takeoff_time = None
    land_mode_observed = False
    position_integral = np.zeros(2)
    last_setpoint_time = None
    process = connection = None
    viewer = None
    pacing_start = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    started = time.monotonic()

    def event(name, **fields):
        row = {"physics_time": float(plant.data.time), "event": name, **fields}
        summary["events"].append(row)
        print(json.dumps(row), flush=True)

    def transition(name):
        nonlocal phase, phase_start, stable_since, takeoff_time
        phase, phase_start, stable_since = name, float(plant.data.time), None
        if name == 'TAKEOFF':
            takeoff_time = phase_start
        if name == "RETURN":
            position_integral[:] = 0
        event(name)

    def command_long(cmd, *args):
        connection.mav.command_long_send(
            1, 1, cmd, 0, *(list(args) + [0] * (7 - len(args)))
        )

    def position_target(target, horizontal_limit=.06, vertical_limit=0.,
                        velocity_feedforward=None, position_gain=.8,
                        vertical_position_gain=None):
        nonlocal next_send, last_setpoint_time
        t = float(plant.data.time)
        if t < next_send:
            return
        if t - telemetry.get("local_time", -100) > 0.5:
            raise RuntimeError("Stale LOCAL_POSITION_NED for precision return")
        measured = np.array(telemetry["local_position"])
        measured += np.array(telemetry["local_velocity"]) * (
            t - telemetry["local_time"]
        )
        position_error = np.array(target) - (plant.qpos[:3] - plant.origin)
        target_ned = measured + to_ap(position_error)
        dt = 0 if last_setpoint_time is None else t - last_setpoint_time
        if phase in ("RETURN", "DESCEND", "ALIGN", "FINAL_DESCEND"):
            position_integral[:] = np.clip(
                position_integral + 0.1 * position_error[:2] * dt, -0.015, 0.015
            )
        desired_velocity = np.zeros(3) if velocity_feedforward is None else np.asarray(velocity_feedforward, float).copy()
        desired_velocity[:2] += position_gain * position_error[:2] + position_integral
        vertical_position_gain = position_gain if vertical_position_gain is None else vertical_position_gain
        desired_velocity[2] += vertical_position_gain * position_error[2]
        desired_velocity[:2] = np.clip(desired_velocity[:2], -horizontal_limit, horizontal_limit)
        desired_velocity[2] = np.clip(desired_velocity[2], -vertical_limit, vertical_limit)
        # Compensate the EKF velocity discrepancy in the commanded feed-forward.
        # This changes setpoints only; the sensor stream and physical state remain real.
        velocity_ff = np.array(telemetry["local_velocity"]) - to_ap(plant.qvel[:3])
        if np.linalg.norm(velocity_ff) > 0.25:
            raise RuntimeError(
                "Estimated/true velocity discrepancy exceeds precision-test bound"
            )
        commanded_velocity = velocity_ff + to_ap(desired_velocity)
        m = mavutil.mavlink
        mask = (
            m.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | m.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | m.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | m.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        connection.mav.set_position_target_local_ned_send(
            int(t * 1000),
            1,
            1,
            m.MAV_FRAME_LOCAL_NED,
            mask,
            *target_ned,
            *commanded_velocity,
            0,
            0,
            0,
            -initial_yaw,
            0,
        )
        next_send = t + 0.05
        last_setpoint_time = t

    try:
        if view:
            from ..simulation.live_view import FlightView

            viewer = FlightView(plant)
            viewer.update(phase, telemetry, force=True)
        # Viewer initialization is outside the SITL startup timeout.
        started = time.monotonic()
        sock.bind(("127.0.0.1", 9002 + 10 * instance))
        sock.setblocking(False)
        with (output / "sitl.log").open("w") as sitl_log, (
            output / "frames.jsonl"
        ).open("w") as frames, (output / "mavlink.jsonl").open("w") as messages:
            process = subprocess.Popen(
                command, cwd=output, stdout=sitl_log, stderr=subprocess.STDOUT
            )
            while connection is None:
                if time.monotonic() - started > 15 or process.poll() is not None:
                    raise RuntimeError("SITL startup failed/timed out")
                try:
                    connection = mavutil.mavlink_connection(
                        f"tcp:127.0.0.1:{5760+10*instance}", source_system=255
                    )
                except OSError:
                    time.sleep(0.05)
            while time.monotonic() - started < max(120, physics_deadline / speedup + 120 if route is not None else 160 / speedup + 30):
                if viewer is not None and not viewer.is_running():
                    raise RuntimeError("Live viewer closed; flight cancelled")
                if process.poll() is not None:
                    raise RuntimeError("SITL exited")
                if not select.select([sock], [], [], 0.1)[0]:
                    continue
                payload, address = sock.recvfrom(1024)
                reply, row = bridge.process(payload)
                sock.sendto(reply, address)
                t = float(plant.data.time)
                if viewer is not None:
                    if pacing_start is None:
                        pacing_start = time.monotonic() - t / speedup
                    delay = pacing_start + t / speedup - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, 0.05))
                while True:
                    msg = connection.recv_match(blocking=False)
                    if msg is None:
                        break
                    kind = msg.get_type()
                    if kind == "BAD_DATA":
                        continue
                    messages.write(
                        json.dumps(
                            {"physics_time": t, "message": msg.to_dict()}, default=str
                        )
                        + "\n"
                    )
                    if kind == "HEARTBEAT":
                        telemetry.update(
                            mode=mavutil.mode_string_v10(msg),
                            armed=bool(
                                msg.base_mode
                                & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                            ),
                            heartbeat_time=t,
                        )
                        if not requested:
                            connection.mav.request_data_stream_send(
                                1, 1, mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1
                            )
                            connection.mav.param_request_list_send(1, 1)
                            requested = True
                    elif kind == "LOCAL_POSITION_NED":
                        telemetry.update(
                            local_position=[msg.x, msg.y, msg.z],
                            local_velocity=[msg.vx, msg.vy, msg.vz],
                            local_time=t,
                        )
                    elif kind == "EKF_STATUS_REPORT":
                        telemetry["ekf_flags"] = msg.flags
                    elif kind == "PARAM_VALUE":
                        params[msg.param_id] = msg.param_value
                    elif kind == "STATUSTEXT":
                        event("STATUSTEXT", text=msg.text)
                        if msg.text == "ArduPilot Ready":
                            telemetry["ready"] = True
                    elif kind == "COMMAND_ACK":
                        event("COMMAND_ACK", command=msg.command, result=msg.result)
                if t >= next_heartbeat:
                    connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0,
                        0,
                        0,
                    )
                    next_heartbeat = t + 1
                if row is None:
                    continue
                local = plant.qpos[:3] - plant.origin
                speed = float(np.linalg.norm(plant.qvel[:3]))
                rotation = plant.data.xmat[plant.body].reshape(3, 3)
                tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
                max_tilt = max(max_tilt, tilt)
                contacts = plant.contacts()
                row.update(
                    phase=phase,
                    telemetry=telemetry.copy(),
                    contacts=contacts,
                    all_qpos=plant.data.qpos.tolist(),
                    all_qvel=plant.data.qvel.tolist(),
                    all_motor_thrust_N=plant.data.actuator_force.tolist(),
                )
                frames.write(json.dumps(row, allow_nan=False) + "\n")
                if viewer is not None:
                    viewer.update(phase, telemetry)
                if not np.isfinite(
                    np.r_[plant.data.qpos, plant.data.qvel, plant.data.qacc]
                ).all():
                    raise RuntimeError("Nonfinite physics state")
                maximum_tilt = math.radians(route.maximum_tilt_deg) if phase == 'ROUTE' else math.radians(25)
                maximum_speed = route.maximum_speed_m_s if phase == 'ROUTE' else 2
                if (
                    tilt > maximum_tilt
                    or (phase != 'ROUTE' and local[2] > 1.1)
                    or (phase != 'ROUTE' and np.linalg.norm(local[:2]) > 0.5)
                    or speed > maximum_speed
                ):
                    raise RuntimeError(
                        f"Flight envelope exceeded: local={local}, speed={speed}, tilt={math.degrees(tilt)}"
                    )
                if phase == 'ROUTE':
                    route.check(plant.qpos[:3])
                    if any(not route.contact_allowed(c) for c in contacts):
                        raise RuntimeError('Unexpected contact on armor route')
                for qa, va, initial in plant.inactive:
                    drift = float(
                        np.linalg.norm(plant.data.qpos[qa : qa + 2] - initial[:2])
                    )
                    max_inactive_drift = max(max_inactive_drift, drift)
                    if (
                        drift > 0.001
                        or abs(plant.data.qpos[qa + 2] - initial[2]) > 0.002
                    ):
                        raise RuntimeError("Inactive vehicle moved/lifted")
                    if t > 1 and np.max(np.abs(plant.data.qvel[va : va + 6])) > 0.001:
                        raise RuntimeError("Inactive vehicle is not resting")
                    q = plant.data.qpos[qa + 3 : qa + 7]
                    if 1 - 2 * (q[1] ** 2 + q[2] ** 2) < math.cos(math.radians(1)):
                        raise RuntimeError("Inactive vehicle tilted")
                inactive_motors = np.setdiff1d(np.arange(plant.model.nu), plant.motors)
                if np.any(plant.data.ctrl[inactive_motors]) or np.any(
                    plant.data.actuator_force[inactive_motors]
                ):
                    raise RuntimeError("Inactive vehicle received thrust")
                if phase in ("OFFSET", "RETURN", "ALIGN") and contacts:
                    raise RuntimeError("Unexpected contact during airborne translation")
                for c in contacts:
                    max_force = max(max_force, c["force_N"])
                    max_penetration = max(max_penetration, -c["distance_m"])
                    if ("rm27_dock_collision_bottom" not in c["geoms"]
                            and not (phase == 'ROUTE' and route.contact_allowed(c))):
                        raise RuntimeError(f"Unintended contact: {c['geoms']}")
                    if ("rm27_dock_collision_bottom" in c["geoms"]
                            and (c["force_N"] > 10 or c["distance_m"] < -0.005)):
                        raise RuntimeError("Unstable landing contact")
                if phase != "INITIALIZE":
                    if t - telemetry.get("heartbeat_time", 0) > 2:
                        raise RuntimeError("Stale armed/mode telemetry")
                    if phase not in ("LAND", "POST") and (
                        not telemetry["armed"] or telemetry["mode"] != "GUIDED"
                    ):
                        raise RuntimeError("Unexpected disarm or mode change")
                if t > physics_deadline:
                    raise RuntimeError("Physics deadline")

                if phase == "INITIALIZE":
                    if t > 50 and not summary["parameters_verified"]:
                        mismatch = {
                            k: {"expected": v, "received": params.get(k)}
                            for k, v in expected.items()
                            if k not in params
                            or not math.isclose(params[k], v, abs_tol=1e-6)
                        }
                        raise RuntimeError(f"Parameter verification failed: {mismatch}")
                    if not summary["parameters_verified"] and all(
                        k in params and math.isclose(params[k], v, abs_tol=1e-6)
                        for k, v in expected.items()
                    ):
                        summary["parameters_verified"] = True
                        event(
                            "parameters_verified",
                            values={k: params[k] for k in expected},
                        )
                    if (
                        (t >= next_action or telemetry["armed"])
                        and summary["parameters_verified"]
                        and telemetry["ready"]
                    ):
                        next_action = t + 2
                        if telemetry["mode"] != "GUIDED":
                            connection.set_mode("GUIDED")
                        elif not telemetry["armed"]:
                            if telemetry["ekf_flags"] & 0x0B == 0x0B:
                                command_long(
                                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1
                                )
                                event("request_ARM")
                        else:
                            event("ARMED")
                            command_long(
                                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0.6,
                            )
                            transition("TAKEOFF")
                elif phase == "TAKEOFF":
                    if abs(local[2] - 0.6) < 0.08 and speed < 0.08:
                        if stable_since is None:
                            stable_since = t
                        settle = route.mission['takeoff_settle_s'] if route is not None else 2
                        if t - stable_since >= settle:
                            transition("HOVER")
                    else:
                        stable_since = None
                    if t - phase_start > 20:
                        raise RuntimeError("Takeoff failed to settle")
                elif phase == "HOVER":
                    position_target([0, 0, 0.6])
                    if contacts or abs(local[2] - 0.6) > 0.1 or speed > 0.15:
                        raise RuntimeError("Hover bound violated")
                    hover_samples.append([t, *local, speed, math.degrees(tilt)])
                    hover_duration = route.mission['pre_route_hover_s'] if route is not None else 10
                    if t - hover_samples[0][0] >= hover_duration:
                        transition("OFFSET" if route is None else 'ROUTE')
                elif phase == 'ROUTE':
                    target = route.target(t, plant.qpos[:3], speed, contacts, plant.qvel[:3])
                    if target is None:
                        summary['route'] = {'passed': True, 'visited': route.visited,
                                            'max_path_error_m': route.max_path_error,
                                            'impact': route.impact}
                        transition('RETURN')
                    else:
                        position_target(target-plant.origin,
                                        route.horizontal_command_limit_m_s,
                                        route.vertical_command_limit_m_s,
                                        route.reference_velocity,
                                        route.mission['truth_position_gain_s_inv'],
                                        route.mission['truth_vertical_position_gain_s_inv'])
                elif phase == "OFFSET":
                    target_x = min(0.12, 0.04 * (t - phase_start))
                    position_target([target_x, 0, 0.6])
                    if (
                        t - phase_start > 6
                        and np.linalg.norm(local - [0.12, 0, 0.6]) < 0.015
                        and speed < 0.02
                    ):
                        event("offset_reached", position_local_nwu=local.tolist())
                        transition("RETURN")
                    if t - phase_start > 20:
                        raise RuntimeError("Offset did not settle")
                elif phase == "RETURN":
                    return_x = 0 if route is not None else max(0, 0.12 - 0.04 * (t - phase_start))
                    position_target([return_x, 0, 0.6])
                    return_wait = route.mission['return_settle_s'] if route is not None else 6
                    return_xy = .015 if route is not None else .004
                    return_speed = .08 if route is not None else .01
                    if (
                        t - phase_start > return_wait
                        and np.linalg.norm(local[:2]) < return_xy
                        and abs(local[2] - 0.6) < 0.015
                        and speed < return_speed
                    ):
                        event("returned_over_slot", position_local_nwu=local.tolist())
                        transition("DESCEND")
                    if t - phase_start > 25:
                        raise RuntimeError("Return alignment failed")
                elif phase == "DESCEND":
                    descend_rate = route.mission['descent_reference_speed_m_s'] if route is not None else .04
                    position_target([0, 0, max(0.12, 0.6 - descend_rate * (t - phase_start))],
                                    vertical_limit=descend_rate)
                    if abs(local[2] - 0.12) < 0.005 and abs(plant.qvel[2]) < 0.01:
                        transition("ALIGN")
                    elif t - phase_start > 25:
                        raise RuntimeError("Approach did not settle above dock rim")
                elif phase == "ALIGN":
                    position_target([0, 0, 0.12])
                    if (
                        np.linalg.norm(local[:2]) < 0.003
                        and np.linalg.norm(plant.qvel[:2]) < 0.006
                        and abs(local[2] - 0.12) < 0.005
                    ):
                        if stable_since is None:
                            stable_since = t
                        align_hold = route.mission['alignment_hold_s'] if route is not None else .75
                        if t - stable_since >= align_hold:
                            transition("FINAL_DESCEND")
                    else:
                        stable_since = None
                    if t - phase_start > 20:
                        raise RuntimeError("Final dock alignment did not remain stable")
                elif phase == "FINAL_DESCEND":
                    final_rate = route.mission['final_descent_reference_speed_m_s'] if route is not None else .03
                    position_target([0, 0, max(-0.02, 0.12 - final_rate * (t - phase_start))],
                                    vertical_limit=final_rate)
                    if local[2] < 0.1 and np.linalg.norm(local[:2]) > 0.006:
                        raise RuntimeError("Dock alignment lost near rim")
                    if contacts:
                        landing = {
                            "physics_time": t,
                            "local_position": local.tolist(),
                            "speed_m_s": speed,
                            "tilt_deg": math.degrees(tilt),
                            "contacts": contacts,
                        }
                        if speed > 0.15:
                            raise RuntimeError("Landing too fast")
                        if route is not None:
                            mission_duration = t-takeoff_time
                            landing['takeoff_to_dock_contact_s'] = mission_duration
                            if mission_duration > route.mission['maximum_takeoff_to_dock_contact_s']:
                                raise RuntimeError(f"30-second mission deadline exceeded: {mission_duration:.3f} s")
                        event("touchdown", **landing)
                        connection.set_mode("LAND")
                        next_action = t + 1
                        transition("LAND")
                    if t - phase_start > 25:
                        raise RuntimeError("Descent failed to contact dock")
                elif phase == "LAND":
                    if telemetry["mode"] == "LAND" and telemetry["armed"]:
                        land_mode_observed = True
                    if telemetry["mode"] not in ("GUIDED", "LAND"):
                        raise RuntimeError("Unexpected landing mode")
                    if t >= next_action and telemetry["mode"] != "LAND":
                        connection.set_mode("LAND")
                        next_action = t + 1
                    if (
                        not contacts
                        or np.linalg.norm(local[:2]) > 0.007
                        or abs(local[2] - 0.02) > 0.005
                    ):
                        raise RuntimeError("Landing did not remain in assigned slot")
                    if not telemetry["armed"]:
                        if not land_mode_observed or telemetry["mode"] != "LAND":
                            raise RuntimeError(
                                "Disarmed without an observed normal LAND sequence"
                            )
                        transition("POST")
                    elif t - phase_start > 15:
                        raise RuntimeError("Normal landing auto-disarm timed out")
                elif phase == "POST":
                    if telemetry["mode"] != "LAND":
                        raise RuntimeError("Unexpected mode after landing")
                    if (
                        telemetry["armed"]
                        or not contacts
                        or speed > 0.01
                        or np.linalg.norm(local[:2]) > 0.007
                        or abs(local[2] - 0.02) > 0.005
                    ):
                        raise RuntimeError("Disarmed resting gate failed")
                    if (
                        t - phase_start >= 1
                        and np.max(np.abs(plant.data.actuator_force[plant.motors]))
                        < 1e-5
                    ):
                        summary["passed"] = True
                        event(
                            "complete",
                            position_local_nwu=local.tolist(),
                            actual_thrust_N=plant.data.actuator_force[
                                plant.motors
                            ].tolist(),
                        )
                        break
            if not summary["passed"]:
                raise RuntimeError("Wall-clock deadline")
    except (Exception, KeyboardInterrupt) as error:
        summary["failure"] = f"{type(error).__name__}: {error}"
        event("failure", reason=summary["failure"])
    finally:
        if connection is not None:
            connection.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sock.close()
        if viewer is not None:
            summary["viewer_updates"] = viewer.updates
            if viewer.is_running():
                viewer.update(
                    "PASS" if summary["passed"] else "STOPPED", telemetry, force=True
                )
            viewer.close()
        if any(bridge.stats[k] for k in ("duplicates", "missing", "resets", "invalid")):
            summary.update(passed=False, failure="Communication integrity gate failed")
        summary.update(
            phase=phase,
            physics_seconds=float(plant.data.time),
            wall_seconds=time.monotonic() - started,
            communication=bridge.stats,
            max_tilt_deg=math.degrees(max_tilt),
            max_contact_force_N=max_force,
            max_penetration_m=max_penetration,
            max_inactive_xy_drift_m=max_inactive_drift,
            final_local_position=(plant.qpos[:3] - plant.origin).tolist(),
            final_telemetry=telemetry,
            landing=landing,
            hover_samples=len(hover_samples),
            final_all_qpos=plant.data.qpos.tolist(),
            final_all_qvel=plant.data.qvel.tolist(),
            final_all_motor_thrust_N=plant.data.actuator_force.tolist(),
        )
        if hover_samples:
            h = np.array(hover_samples)
            summary["hover"] = {
                "duration_s": float(h[-1, 0] - h[0, 0]),
                "height_min_m": float(h[:, 3].min()),
                "height_max_m": float(h[:, 3].max()),
                "speed_max_m_s": float(h[:, 4].max()),
                "tilt_max_deg": float(h[:, 5].max()),
            }
        (output / "parameters.json").write_text(json.dumps(params, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=31)
    parser.add_argument("--dynamics", type=Path, help="Must match the compiled scene; simulation only")
    parser.add_argument("--view", action="store_true", help="Watch the live flight")
    parser.add_argument(
        "--speedup",
        type=float,
        help="Requested speed: default 1 with --view, otherwise 5",
    )
    args = parser.parse_args()
    return run(
        args.scene, args.output, args.instance, view=args.view, speedup=args.speedup, dynamics=args.dynamics
    )


if __name__ == "__main__":
    raise SystemExit(main())
