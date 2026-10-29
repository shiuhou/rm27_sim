"""Monte Carlo of ideal terminal bottom impacts followed by post-impact recovery.

Every condition sees the same pre-impact cases and the same physical draws (paired).
Results are simulation evidence only; contact compliance and thrust are not calibrated.
"""
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from rm27_sim.paths import ROOT, resolve_asset

try:
    import rm27_contact_physics  # noqa: F401
except ImportError:  # Source checkout without installing packages/physics.
    sys.path.insert(0, str(ROOT / "packages/physics/src"))

import mujoco
from rm27_contact_physics.impact import ImpactCase, RecoverySpec, SuccessSpec, run_trial
from rm27_contact_physics.plant import RM27Plant
from rm27_contact_physics.recovery import RecoveryGains

SOURCES = ("experiment_config", "vehicle_config", "dynamics_profile", "target_geometry")


def sample_case(rng, ranges, fixed):
    def uniform(key):
        lo, hi = ranges[key]
        return float(rng.uniform(lo, hi))

    def planar(magnitude):
        angle = rng.uniform(0, 2 * np.pi)
        return (magnitude * np.cos(angle), magnitude * np.sin(angle))

    omega = rng.normal(size=3)
    omega *= uniform("omega_norm_rad_s") / np.linalg.norm(omega)
    values = dict(
        normal_speed_m_s=uniform("normal_speed_m_s"),
        tangent_velocity_m_s=planar(uniform("tangent_speed_m_s")),
        tilt_deg=uniform("tilt_deg"),
        tilt_azimuth_deg=float(rng.uniform(0, 360)),
        yaw_deg=uniform("yaw_deg"),
        omega_body_rad_s=tuple(omega.tolist()),
        # Square-root radius gives a uniform draw over the offset disk.
        tangent_offset_m=planar(ranges["tangent_offset_m"][1] * np.sqrt(rng.uniform())),
    )
    values.update(fixed)
    return ImpactCase(**values)


def recovery_spec(entry):
    entry = dict(entry)
    gains = RecoveryGains(**entry.pop("gains", {}))
    return RecoverySpec(**entry, gains=gains)


def grid_cells(grid):
    cells = [{}]
    for key, values in grid.items():
        cells = [dict(cell, **{key: value}) for cell in cells for value in values]
    return cells


def summarize(rows):
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    recovered = [r for r in rows if r["outcome"] == "RECOVERED"]

    def quantiles(key, items):
        values = [r[key] for r in items if r[key] is not None]
        return None if not values else dict(zip(("p10", "p50", "p90", "max"), np.round(
            [*np.percentile(values, [10, 50, 90]), np.max(values)], 4).tolist()))

    k = len(recovered)
    n = len(rows)
    # Wilson 95% interval for the recovery rate.
    z = 1.96
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return dict(trials=n, recovered=k, recovery_rate=k / n,
                recovery_rate_wilson95=[round(centre - half, 4), round(centre + half, 4)],
                outcomes=dict(sorted(outcomes.items())),
                bottom_only_fraction=float(np.mean([r["bottom_only"] for r in rows])),
                recovery_time_s=quantiles("recovery_time_s", recovered),
                altitude_loss_m=quantiles("altitude_loss_m", rows),
                max_distance_m=quantiles("max_distance_m", rows),
                contact_time_s=quantiles("contact_time_s", rows),
                motor_saturation_time_s=quantiles("motor_saturation_time_s", rows),
                peak_specific_force_g=quantiles("peak_specific_force_g", rows),
                peak_normal_force_N=quantiles("peak_normal_force_N", rows))


def provenance(config_path, cfg, sources):
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    code = sorted((ROOT / "packages/physics/src").rglob("*.py")) + sorted((ROOT / "rm27_sim").rglob("*.py"))
    return dict(config=str(config_path), config_sha256=sha(config_path),
                sources={k: dict(path=str(v), sha256=sha(v)) for k, v in sources.items()},
                git_head=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                mujoco=mujoco.__version__, numpy=np.__version__, python=platform.python_version(),
                seed=cfg["seed"], physical_scale=cfg["physical_scale"],
                source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
                sampling="completed_interval_solve_delivered_at_endpoint_v1",
                limitations="Ideal pre-impact states and truth state feedback; MuJoCo soft contact and "
                            "thrust/inertia are uncalibrated engineering estimates; not hardware evidence.")


def load_study(config_path):
    """Resolve one study config into everything needed to rebuild its plants and trials."""
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("Impact recovery config requires schema_version 1")
    sources = {name: (ROOT / cfg[name]).resolve() for name in SOURCES}
    if "vehicle_mjcf" in cfg:  # runtime aircraft MJCF, resolved through the registered mjcf asset root
        sources["vehicle_mjcf"] = resolve_asset(cfg["vehicle_mjcf"]).resolve()
    if "rate_params" in cfg:
        sources["rate_params"] = (ROOT / cfg["rate_params"]).resolve()
    if "imu_params" in cfg:
        sources["imu_params"] = (ROOT / cfg["imu_params"]).resolve()
    base = json.loads(sources["experiment_config"].read_text())
    # Study-level overrides: armor contact parameters, uncertainty ranges, bottom contact geometry.
    base["target"].update(cfg.get("target_overrides", {}))
    base["uncertainty"].update(cfg.get("uncertainty_overrides", {}))
    base["contact"].update(cfg.get("contact_overrides", {}))
    if "physics_dt_s" in cfg:  # study-level override, e.g. for IMU sampling above 1 kHz
        base["physics_dt_s"] = float(cfg["physics_dt_s"])
    return dict(
        path=config_path, cfg=cfg, sources=sources, base=base,
        sources_plant={k: sources[k] for k in ("vehicle_config", "dynamics_profile", "target_geometry", "vehicle_mjcf")
                       if k in sources},
        conditions={name: recovery_spec(entry) for name, entry in cfg["conditions"].items()},
        success=SuccessSpec(**cfg.get("success", {})),
        rate_params=json.loads(sources["rate_params"].read_text()) if "rate_params" in sources else None,
        imu_params=json.loads(sources["imu_params"].read_text()) if "imu_params" in sources else None)


def sampled_cases(cfg):
    """One deterministic case/physical draw stream, independent of conditions/workers."""
    rng = np.random.default_rng(cfg["seed"])
    for cell_index, cell in enumerate(grid_cells(cfg.get("grid", {}))):
        for repeat in range(cfg["trials_per_cell"]):
            case = sample_case(rng, cfg["sampling"], dict(cfg.get("fixed", {}), **cell))
            seed = int(rng.integers(2**31)) if cfg["physical_scale"] > 0 else None
            yield cell_index, cell, repeat, case, seed


def _run_case(study, payload):
    cell_index, cell, repeat, case, physical_seed = payload
    plant = RM27Plant(study["base"], study["sources_plant"], physical_seed=physical_seed,
                      physical_scale=study["cfg"]["physical_scale"],
                      physical_values=study["cfg"].get("physical_values"))
    return [dict(condition=name, cell=cell, cell_index=cell_index, repeat=repeat,
                 physical_seed=physical_seed, case=dataclasses.asdict(case),
                 **run_trial(plant, case, spec, study["success"], rate_params=study["rate_params"],
                             imu_params=study["imu_params"], imu_seed=physical_seed))
            for name, spec in study["conditions"].items()]


def run(config_path, output, *, workers=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    study = load_study(config_path)
    config_path, cfg, sources, base = study["path"], study["cfg"], study["sources"], study["base"]
    sources_plant, conditions, success = study["sources_plant"], study["conditions"], study["success"]
    rate_params, imu_params = study["rate_params"], study["imu_params"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    output.mkdir(parents=True)
    rows = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        execute = partial(_run_case, study)
        results = (pool.map(execute, sampled_cases(cfg), chunksize=4) if pool
                   else map(execute, sampled_cases(cfg)))
        with (output / "trials.jsonl").open("w") as handle:
            for batch in results:
                for row in batch:
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    summary = dict(
        overall={name: summarize([r for r in rows if r["condition"] == name]) for name in conditions},
        by_cell=[dict(cell=cell, **{name: summarize([r for r in rows if r["condition"] == name
                                                     and r["cell_index"] == i]) for name in conditions})
                 for i, cell in enumerate(grid_cells(cfg.get("grid", {})))],
        success_definition=dataclasses.asdict(success),
        conditions={name: dataclasses.asdict(spec) for name, spec in conditions.items()},
        provenance=provenance(config_path, cfg, sources),
        effective_config=cfg, effective_experiment=base,
        input_snapshots={k: json.loads(v.read_text()) for k, v in sources.items() if v.suffix == '.json'},
        workers=workers,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
