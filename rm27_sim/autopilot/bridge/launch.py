"""Run an isolated real SITL/MuJoCo ARM→TAKEOFF→10 second hover benchmark."""

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
from .bridge import Bridge

from rm27_sim.paths import ROOT, resolve_asset
EXPECTED_PARAMS = {
    "FRAME_CLASS": 1,
    "FRAME_TYPE": 1,
    "SIM_RATE_HZ": 500,
    "SCHED_LOOP_RATE": 250,
    "MOT_PWM_MIN": 1000,
    "MOT_PWM_MAX": 2000,
    "MOT_THST_EXPO": 0,
    "MOT_BAT_VOLT_MIN": 0,
    "MOT_BAT_VOLT_MAX": 0,
    "MOT_SPIN_MIN": 0.15,
    "MOT_SPIN_MAX": 0.95,
    "AHRS_EKF_TYPE": 3,
    "EK3_ENABLE": 1,
    "ARMING_SKIPCHK": 0,
    "SERVO1_FUNCTION": 33,
    "SERVO2_FUNCTION": 34,
    "SERVO3_FUNCTION": 35,
    "SERVO4_FUNCTION": 36,
}


def verify_parameters(params):
    return all(
        name in params and math.isclose(params[name], value, abs_tol=1e-6)
        for name, value in EXPECTED_PARAMS.items()
    )


def run(
    ap: Path, output: Path, instance=20, speedup=5, *, bridge=None, after_hover=None
):
    output.mkdir(parents=True, exist_ok=False)
    binary = ap / "build/sitl/bin/arducopter"
    defaults = ",".join(
        str(p)
        for p in (
            ap / "Tools/autotest/default_params/copter.parm",
            ap / "Tools/autotest/default_params/copter-X.parm",
            Path(__file__).resolve().parent / "hover.parm",
        )
    )
    command = [
        str(binary),
        "--model",
        "JSON:127.0.0.1",
        "--rate",
        "500",
        "--speedup",
        str(speedup),
        "--instance",
        str(instance),
        "--defaults",
        defaults,
        "--home",
        "-35.363261,149.165230,584,0",
        "--wipe",
    ]
    metadata = {
        "command": command,
        "baseline": "2be0029",
        "ardupilot_commit": subprocess.check_output(
            ["git", "-C", str(ap), "rev-parse", "HEAD"], text=True
        ).strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    bridge = Bridge() if bridge is None else bridge
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9002 + 10 * instance))
    sock.setblocking(False)
    process = None
    connection = None
    summary = {"passed": False, "failure": None, "events": []}
    telemetry = {
        "mode": None,
        "armed": False,
        "altitude_m": None,
        "ekf_flags": 0,
        "ready": False,
    }
    params = {}
    armed_once = False
    takeoff_sent = False
    parameters_verified = False
    armed_reported = False
    hovering_since = None
    settled_since = None
    next_heartbeat = 0
    next_action = 15
    started = time.monotonic()
    max_tilt = 0
    max_alt = 0
    hover_rows = []
    hover_finished_at = None

    def event(name, **fields):
        row = {"physics_time": float(bridge.plant.data.time), "event": name, **fields}
        summary["events"].append(row)
        print(json.dumps(row), flush=True)

    try:
        with (output / "sitl.log").open("w") as sitl_log, (
            output / "frames.jsonl"
        ).open("w") as frames, (output / "mavlink.jsonl").open("w") as messages:
            process = subprocess.Popen(
                command, cwd=output, stdout=sitl_log, stderr=subprocess.STDOUT
            )
            while connection is None:
                if time.monotonic() - started > 15:
                    raise RuntimeError("SITL TCP startup timed out")
                if process.poll() is not None:
                    raise RuntimeError("SITL exited during startup")
                try:
                    connection = mavutil.mavlink_connection(
                        f"tcp:127.0.0.1:{5760+10*instance}", source_system=255
                    )
                except OSError:
                    time.sleep(0.05)  # connection retry only; never advances physics
            requested = False
            while time.monotonic() - started < 120:
                if process.poll() is not None:
                    raise RuntimeError("SITL exited unexpectedly")
                ready, _, _ = select.select([sock], [], [], 0.1)
                if not ready:
                    continue
                payload, address = sock.recvfrom(1024)
                reply, row = bridge.process(payload)
                sock.sendto(reply, address)
                t = bridge.plant.data.time
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
                        telemetry["mode"] = mavutil.mode_string_v10(msg)
                        telemetry["armed"] = bool(
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        )
                        telemetry["heartbeat_time"] = t
                        if not requested:
                            connection.mav.request_data_stream_send(
                                1, 1, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
                            )
                            connection.mav.param_request_list_send(1, 1)
                            requested = True
                    elif kind == "GLOBAL_POSITION_INT":
                        telemetry["altitude_m"] = msg.relative_alt / 1000
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

                def command_long(cmd, *args):
                    connection.mav.command_long_send(
                        1, 1, cmd, 0, *(list(args) + [0] * (7 - len(args)))
                    )

                if verify_parameters(params) and not parameters_verified:
                    parameters_verified = True
                    event(
                        "parameters_verified",
                        values={k: params[k] for k in EXPECTED_PARAMS},
                    )
                if telemetry["armed"] and not armed_reported:
                    armed_reported = True
                    armed_once = True
                    event("ARMED")
                if (
                    t >= next_action
                    and not takeoff_sent
                    and parameters_verified
                    and telemetry["ready"]
                ):
                    next_action = t + 2
                    if telemetry["mode"] != "GUIDED":
                        connection.set_mode("GUIDED")
                        event("request_GUIDED")
                    elif not telemetry["armed"]:
                        # Require attitude, horizontal velocity and absolute horizontal position.
                        if telemetry["ekf_flags"] & 0x0B == 0x0B:
                            command_long(
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1
                            )
                            event("request_ARM")
                    else:
                        armed_once = True
                        command_long(
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 1.5
                        )
                        takeoff_sent = True
                        event("request_TAKEOFF", altitude_m=1.5)
                if row is None:
                    continue
                row["ardupilot"] = telemetry.copy()
                frames.write(json.dumps(row, allow_nan=False) + "\n")
                p = bridge.plant
                altitude = -float(row["state_sent"]["position"][2])
                horizontal_distance = float(
                    np.linalg.norm(row["state_sent"]["position"][:2])
                )
                tilt = math.acos(
                    float(np.clip(p.data.xmat[p.body].reshape(3, 3)[2, 2], -1, 1))
                )
                max_tilt = max(max_tilt, tilt)
                max_alt = max(max_alt, altitude)
                if not np.all(
                    np.isfinite(np.r_[p.data.qpos, p.data.qvel, p.data.qacc])
                ):
                    raise RuntimeError("nonfinite truth")
                if tilt > math.radians(45) or altitude > 5 or horizontal_distance > 10:
                    raise RuntimeError("flip, runaway altitude or horizontal runaway")
                if armed_once and t - telemetry.get("heartbeat_time", 0) > 2:
                    raise RuntimeError("stale armed/mode telemetry")
                if armed_once and not telemetry["armed"]:
                    raise RuntimeError("unexpected disarm")
                if takeoff_sent and telemetry["mode"] != "GUIDED":
                    raise RuntimeError("unexpected mode change")
                if t > 120:
                    raise RuntimeError("physics deadline before experiment completed")
                if hover_finished_at is not None:
                    if after_hover(connection, bridge, telemetry, event):
                        summary["passed"] = True
                        event("experiment_complete")
                        break
                    continue
                if (
                    takeoff_sent
                    and abs(altitude - 1.5) < 0.3
                    and abs(p.data.qvel[2]) < 0.3
                ):
                    if settled_since is None:
                        settled_since = t
                    if t - settled_since < 2:
                        continue
                    if hovering_since is None:
                        hovering_since = t
                        event("hover_window_start")
                    hover_rows.append(
                        [
                            t,
                            altitude,
                            float(np.linalg.norm(p.data.qvel)),
                            tilt,
                            horizontal_distance,
                        ]
                    )
                    if t - hovering_since >= 10:
                        hover_finished_at = t
                        event("hover_complete")
                        if after_hover is None:
                            summary["passed"] = True
                            break
                else:
                    settled_since = None
                    hovering_since = None
                    hover_rows = []
                if t > 120:
                    raise RuntimeError("physics deadline before hover completed")
            if not summary["passed"]:
                raise RuntimeError("wall-clock deadline before hover completed")
    except Exception as error:
        summary["failure"] = f"{type(error).__name__}: {error}"
        event("failure", reason=summary["failure"])
    finally:
        if connection:
            connection.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sock.close()
        summary.update(
            {
                "physics_time": float(bridge.plant.data.time),
                "communication": bridge.stats,
                "frame_rates": sorted(bridge.rates),
                "armed": armed_once,
                "takeoff_sent": takeoff_sent,
                "parameters_verified": parameters_verified,
                "max_altitude_m": max_alt,
                "max_tilt_deg": math.degrees(max_tilt),
                "final_telemetry": telemetry,
                "hover_duration_s": (
                    0
                    if hovering_since is None
                    else float(
                        (
                            hover_finished_at
                            if hover_finished_at is not None
                            else bridge.plant.data.time
                        )
                        - hovering_since
                    )
                ),
            }
        )
        if hover_rows:
            a = np.asarray(hover_rows)
            summary["hover"] = {
                "altitude_min_m": float(a[:, 1].min()),
                "altitude_max_m": float(a[:, 1].max()),
                "speed_max_m_s": float(a[:, 2].max()),
                "tilt_max_deg": float(np.degrees(a[:, 3]).max()),
                "horizontal_distance_max_m": float(a[:, 4].max()),
            }
        if bridge.stats["missing"] or bridge.stats["resets"] or bridge.stats["invalid"]:
            summary["passed"] = False
            summary["failure"] = "communication gaps/resets/invalid packets"
        (output / "parameters.json").write_text(
            json.dumps(params, indent=2, sort_keys=True)
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ardupilot", type=Path, default=resolve_asset("rm27/autopilot/vendor/ardupilot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=20)
    parser.add_argument("--speedup", type=float, default=5)
    args = parser.parse_args()
    return run(
        args.ardupilot.resolve(), args.output.resolve(), args.instance, args.speedup
    )


if __name__ == "__main__":
    raise SystemExit(main())
