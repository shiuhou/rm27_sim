"""Replay logged motor inputs exactly: python -m rm27.simulation.replay_micro_flight RUN_DIR."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def maximum_error(actual, expected):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Replay state schema differs from recorded state")
        return max(
            (maximum_error(actual[k], value) for k, value in expected.items()),
            default=0.0,
        )
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Replay state array length changed")
        return max((maximum_error(a, b) for a, b in zip(actual, expected)), default=0.0)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError("Nonfinite state in replay comparison")
    return abs(actual - expected)


def replay(run_dir, render=False):
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = {
        "passed": False,
        "original_flight_passed": bool(summary["passed"]),
        "original_flight_failure": summary.get("failure"),
        "scope": "Recorded-PWM physics replay; does not repeat the closed-loop SITL flight",
        "compared_frames": 0,
        "compared_full_scene_frames": 0,
        "exactly_matching_frames": 0,
        "maximum_absolute_state_error": 0.0,
        "maximum_absolute_truth_error": 0.0,
        "maximum_absolute_full_scene_error": 0.0,
        "final_full_scene_verified": False,
        "failure": None,
    }
    original_state = hashlib.sha256()
    replayed_state = hashlib.sha256()
    original_truth = hashlib.sha256()
    replayed_truth = hashlib.sha256()
    original_full_scene = hashlib.sha256()
    replayed_full_scene = hashlib.sha256()
    full_scene_fields = ("all_qpos", "all_qvel", "all_motor_thrust_N")
    renderer = None
    captures = []
    seen_phases = set()
    try:
        scene = Path(metadata["scene"]).resolve()
        scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
        result["scene_sha256"] = scene_sha256
        if scene_sha256 != metadata["scene_sha256"]:
            raise ValueError("Scene XML changed since the recorded flight")
        communication = summary["communication"]
        if any(
            communication[k] for k in ("duplicates", "missing", "resets", "invalid")
        ):
            raise ValueError(
                "Original flight has communication gaps, duplicates, resets or invalid packets"
            )
        if render:
            os.environ.setdefault("MUJOCO_GL", "egl")
        from .micro_plant import MicroPlant

        vehicle_name = metadata["vehicle"]
        if vehicle_name not in ("drone_1", "drone_2", "drone_3", "drone_4"):
            raise ValueError("Unknown selected vehicle in flight metadata")
        dynamics = None
        if metadata.get('dynamics_profile'):
            dynamics = run_dir / metadata['dynamics_profile']
            if hashlib.sha256(dynamics.read_bytes()).hexdigest() != metadata['dynamics_profile_sha256']:
                raise ValueError('Dynamics profile changed since flight')
        plant = MicroPlant(scene, vehicle=int(vehicle_name.removeprefix("drone_")), dynamics_profile=dynamics)
        if plant.origin.tolist() != metadata["origin_nwu"]:
            raise ValueError("Selected slot origin changed since recorded flight")
        if render:
            import mujoco
            from PIL import Image, ImageDraw

            renderer = mujoco.Renderer(plant.model, height=640, width=960)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = plant.origin + [0, 0, 0.25]
            camera.distance = 1.3
            camera.azimuth = 45
            camera.elevation = -30
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0

            def capture(label):
                renderer.update_scene(plant.data, camera=camera, scene_option=options)
                image = Image.fromarray(renderer.render())
                captures.append((label, float(plant.data.time), image))

        previous_count = None
        last_phase = None
        with (run_dir / "frames.jsonl").open() as frames:
            for line in frames:
                row = json.loads(line)
                count = row["frame_count"]
                expected_count = (
                    0 if previous_count is None else (previous_count + 1) % (2**32)
                )
                if count != expected_count or row["epoch"] != 0:
                    raise ValueError(
                        f"Non-contiguous original frame history at frame {count}"
                    )
                if row["frame_rate"] != 500 and not (
                    count == 0 and previous_count is None and row["frame_rate"] == 1200
                ):
                    raise ValueError(f"Unexpected frame rate at frame {count}")
                state = plant.step(row["pwm"])
                truth = plant.truth()
                result["compared_frames"] += 1
                result["maximum_absolute_state_error"] = max(
                    result["maximum_absolute_state_error"],
                    maximum_error(state, row["state_sent"]),
                )
                result["maximum_absolute_truth_error"] = max(
                    result["maximum_absolute_truth_error"],
                    maximum_error(truth, row["truth"]),
                )
                original_state.update(encoded(row["state_sent"]))
                replayed_state.update(encoded(state))
                original_truth.update(encoded(row["truth"]))
                replayed_truth.update(encoded(truth))
                if state != row["state_sent"] or truth != row["truth"]:
                    result["first_diverging_frame"] = count
                    raise ValueError(f"Physics replay diverged at frame {count}")
                if any(key in row for key in full_scene_fields):
                    if not all(key in row for key in full_scene_fields):
                        raise ValueError(
                            f"Incomplete full-scene recording at frame {count}"
                        )
                    recorded_full = {key: row[key] for key in full_scene_fields}
                    replayed_full = {
                        "all_qpos": plant.data.qpos.tolist(),
                        "all_qvel": plant.data.qvel.tolist(),
                        "all_motor_thrust_N": plant.data.actuator_force.tolist(),
                    }
                    result["compared_full_scene_frames"] += 1
                    result["maximum_absolute_full_scene_error"] = max(
                        result["maximum_absolute_full_scene_error"],
                        maximum_error(replayed_full, recorded_full),
                    )
                    original_full_scene.update(encoded(recorded_full))
                    replayed_full_scene.update(encoded(replayed_full))
                    if replayed_full != recorded_full:
                        result["first_diverging_frame"] = count
                        raise ValueError(
                            f"Full-scene physics replay diverged at frame {count}"
                        )
                result["exactly_matching_frames"] += 1
                previous_count = count
                last_phase = row["phase"]
                if (
                    render
                    and last_phase
                    in ("HOVER", "OFFSET", "RETURN", "ALIGN", "LAND", "POST")
                    and last_phase not in seen_phases
                ):
                    capture(last_phase)
                    seen_phases.add(last_phase)
        if not result["compared_frames"]:
            raise ValueError("Original flight has no recorded physics frames")
        if result["compared_frames"] != communication["frames"]:
            raise ValueError(
                "Frame log length disagrees with original communication summary"
            )
        if plant.data.time != summary["physics_seconds"]:
            raise ValueError(
                "Replay final time differs from the original flight summary"
            )
        full_count = result["compared_full_scene_frames"]
        if full_count and full_count != result["compared_frames"]:
            raise ValueError("Full-scene recording is missing from some physics frames")
        if any(f"final_{key}" in summary for key in full_scene_fields):
            if not all(f"final_{key}" in summary for key in full_scene_fields):
                raise ValueError("Incomplete full-scene final summary")
            actual_final = {
                "all_qpos": plant.data.qpos.tolist(),
                "all_qvel": plant.data.qvel.tolist(),
                "all_motor_thrust_N": plant.data.actuator_force.tolist(),
            }
            expected_final = {key: summary[f"final_{key}"] for key in full_scene_fields}
            if actual_final != expected_final:
                raise ValueError(
                    "Replay full scene differs from the final flight summary"
                )
            result["final_full_scene_verified"] = True
        if (
            original_state.digest() != replayed_state.digest()
            or original_truth.digest() != replayed_truth.digest()
            or original_full_scene.digest() != replayed_full_scene.digest()
        ):
            raise ValueError("Canonical state/truth hashes differ")
        result["passed"] = True
        if render:
            if not captures:
                capture(f"Final recorded frame: {last_phase}")
            columns = min(2, len(captures))
            canvas = Image.new(
                "RGB",
                (960 * columns, 676 * ((len(captures) + columns - 1) // columns)),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            for i, (phase, timestamp, image) in enumerate(captures):
                x, y = i % columns * 960, i // columns * 676
                canvas.paste(image, (x, y + 36))
                draw.text(
                    (x + 12, y + 10),
                    f"{phase} | t = {timestamp:.3f} s | original flight {'PASS' if summary['passed'] else 'FAIL'}",
                    fill="black",
                )
            output = run_dir / "replay-views.png"
            canvas.save(output)
            result["rendered_montage"] = str(output)
            result["rendered_frames"] = [
                {"phase": phase, "physics_time": timestamp}
                for phase, timestamp, _ in captures
            ]
    except Exception as error:
        result.update(passed=False, failure=f"{type(error).__name__}: {error}")
    finally:
        if renderer is not None:
            renderer.close()
        result.update(
            original_states_sha256=original_state.hexdigest(),
            replayed_states_sha256=replayed_state.hexdigest(),
            original_truth_sha256=original_truth.hexdigest(),
            replayed_truth_sha256=replayed_truth.hexdigest(),
            original_full_scene_sha256=(
                original_full_scene.hexdigest()
                if result["compared_full_scene_frames"]
                else None
            ),
            replayed_full_scene_sha256=(
                replayed_full_scene.hexdigest()
                if result["compared_full_scene_frames"]
                else None
            ),
            hash_encoding="sorted-key compact JSON, one newline after every frame",
        )
        (run_dir / "replay.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render replayed phase transitions with MuJoCo",
    )
    args = parser.parse_args()
    result = replay(args.run_dir, render=args.render)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
