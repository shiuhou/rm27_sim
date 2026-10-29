"""Ideal bottom-impact scenario, recovery controller and S1 Monte Carlo runner."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/physics/src"))

from rm27_contact_physics.impact import ImpactCase, RecoverySpec, initial_state, run_trial  # noqa: E402
from rm27_contact_physics.plant import RM27Plant  # noqa: E402
from rm27_contact_physics.model import VehicleEstimate  # noqa: E402
from rm27_contact_physics.recovery import ImpactDetector, RecoveryController  # noqa: E402
from rm27_sim.paths import resolve_asset  # noqa: E402

CONFIGS = ROOT / "configs"
SOURCES = dict(vehicle_config=CONFIGS / "drone.json",
               dynamics_profile=CONFIGS / "impact_recovery/dynamics_150g.json",
               target_geometry=CONFIGS / "base_target.json")


def mission_inputs():
    from rm27_sim.experiments.impact_recovery import load_study
    cfg = json.loads((CONFIGS / "impact_recovery/mission.json").read_text())
    return load_study(ROOT / cfg["base_study"]), cfg


def test_continuous_mission_has_one_reset_and_seeded_replay(monkeypatch):
    from rm27_sim.experiments import impact_mission as mission
    study, cfg = mission_inputs()
    original_reset = RM27Plant.reset
    original_advance = mission.advance_physics_interval
    resets, previous = [], []

    def reset(self, **kwargs):
        resets.append(True)
        previous.clear()
        return original_reset(self, **kwargs)

    def advance(model, data, state, **kwargs):
        # A phase change must not teleport pose, clear velocity or erase motor lag.
        if previous:
            np.testing.assert_array_equal(np.r_[data.qpos, data.qvel, data.act], previous[0])
        result = original_advance(model, data, state, **kwargs)
        previous[:] = [np.r_[data.qpos, data.qvel, data.act].copy()]
        return result

    monkeypatch.setattr(RM27Plant, "reset", reset)
    monkeypatch.setattr(mission, "advance_physics_interval", advance)
    first, trace, _ = mission.simulate(study, cfg, seed=87131)
    assert len(resets) == 1 and first["passed"], first["failure"]
    assert [e["phase"] for e in first["events"]] == list(mission.PHASES)
    assert first["impact"]["valid"] and first["impact"]["bottom_contact"]
    assert first["detection_time_s"] >= first["impact"]["time_s"]
    assert first["recovery"]["recovery_time_s"] <= cfg["mission"]["recovery_deadline_s"]
    assert first["landing"]["time_s"] > first["impact"]["time_s"]
    assert max(abs(x) for x in first["final_motor_force_N"]) < cfg["mission"]["stopped_motor_force_max_N"]
    assert first["final_motor_command"] == [0.]*4
    second, replay, _ = mission.simulate(study, cfg, seed=87131)
    assert len(resets) == 2 and first == second
    np.testing.assert_array_equal(trace, replay)


def test_mission_cannot_recover_or_succeed_when_plate_collision_is_absent(monkeypatch):
    from rm27_sim.experiments import impact_mission as mission
    study, cfg = mission_inputs()
    make = mission.make_plant

    def without_plate(*args, **kwargs):
        plant = make(*args, **kwargs)
        for name in mission.TARGET_GEOMS:
            index = plant.model.geom(name).id
            plant.model.geom_contype[index] = 0
            plant.model.geom_conaffinity[index] = 0
        return plant

    monkeypatch.setattr(mission, "make_plant", without_plate)
    result, _, _ = mission.simulate(study, cfg)
    assert not result["passed"] and result["failure"] == "NO_IMPACT_DETECTION"
    assert result["impact"] is None and result["landing"] is None
    assert all(e["phase"] not in ("RECOVER", "RETURN", "COMPLETE") for e in result["events"])


def test_mission_does_not_return_before_recovery_hold_is_met():
    from dataclasses import replace
    from rm27_sim.experiments.impact_mission import simulate
    study, cfg = mission_inputs()
    study["success"] = replace(study["success"], hold_s=3.)
    result, _, _ = simulate(study, cfg)
    assert not result["passed"] and result["failure"].startswith("RECOVERY_")
    assert result["impact"]["valid"] and result["recovery"] is None
    assert all(e["phase"] != "RETURN" for e in result["events"])


@pytest.mark.parametrize("seed", [1630132026, 1387383711, 1181087528])
def test_mission_handles_previously_failed_thrust_bias_and_terminal_tracking(seed):
    from rm27_sim.experiments.impact_mission import simulate
    study, cfg = mission_inputs()
    result, _, _ = simulate(study, cfg, seed=seed)
    assert result["passed"], result["failure"]
    assert result["impact"]["attitude_error_deg"] <= cfg["mission"]["contact_attitude_error_max_deg"]
    assert result["landing"] is not None
    # Evaluation may inspect the hidden draw; the observer consumes only states
    # and issued motor commands. Its estimate should explain the eliminated bias.
    physical_ratio = result["physical_draw"]["thrust_scale"] / result["physical_draw"]["mass_scale"]
    assert result["inferred_specific_thrust_ratio"] == pytest.approx(physical_ratio, abs=.02)


def test_mission_requires_physical_support_for_landing(monkeypatch):
    from rm27_sim.experiments import impact_mission as mission
    study, cfg = mission_inputs()
    original = mission.advance_physics_interval

    def remove_pad_on_return(model, data, state, **kwargs):
        if state.timestamp > 13.:
            for i in range(model.ngeom):
                if model.geom(i).name.startswith("landing_pad_"):
                    model.geom_contype[i] = model.geom_conaffinity[i] = 0
        return original(model, data, state, **kwargs)

    monkeypatch.setattr(mission, "advance_physics_interval", remove_pad_on_return)
    result, _, _ = mission.simulate(study, cfg)
    assert not result["passed"] and result["failure"] == "GROUND_CONTACT"
    assert result["impact"]["valid"] and result["recovery"] is not None
    assert result["landing"] is None and result["phase"] == "LAND"


def plant(**kwargs):
    base = json.loads((CONFIGS / "impact_recovery/contact_base.json").read_text())
    return RM27Plant(base, SOURCES, **kwargs)


def test_ideal_state_places_bumper_gap_along_plate_normal():
    p = plant()
    n = np.asarray(p.physical["target"]["normal"]) / np.linalg.norm(p.physical["target"]["normal"])
    position, rotation, velocity, _ = initial_state(p, ImpactCase(normal_speed_m_s=2.25, gap_m=0.02))
    bumper = position + rotation @ p.cfg["contact"]["offset_body_m"]
    gap = (bumper - p.physical["target"]["center_m"]) @ n - p.cfg["contact"]["radius_m"]
    assert gap == pytest.approx(0.02, abs=1e-12)
    assert rotation[:, 2] @ n == pytest.approx(1.)
    assert velocity == pytest.approx(-2.25 * n)
    level = initial_state(p, ImpactCase(attitude_mode="level"))[1]
    assert level[:, 2] == pytest.approx([0., 0., 1.])


def test_nominal_ideal_impact_recovers_only_with_controller_and_is_deterministic():
    p = plant()
    first = run_trial(p, ImpactCase(), RecoverySpec(mode="stabilize"))
    again = run_trial(p, ImpactCase(), RecoverySpec(mode="stabilize"))
    assert first == again
    assert first["outcome"] == "RECOVERED" and first["bottom_only"]
    assert first["detection_delay_s"] == pytest.approx(0., abs=2e-3)
    assert run_trial(p, ImpactCase(), RecoverySpec(mode="none"))["outcome"] != "RECOVERED"


def test_controller_hovers_at_setpoint_and_re_erects_when_inverted():
    controller = RecoveryController(0.15, 4.4, setpoint=[0., 0., 1.])
    upright = VehicleEstimate(np.array([0., 0., 1.]), np.zeros(3), np.eye(3), np.zeros(3), 0.)
    assert controller.command(upright) == pytest.approx([0.15 * 9.81, 0., 0., 0.])
    flipped = np.diag([1., -1., -1.])
    command = controller.command(VehicleEstimate(np.array([0., 0., 1.]), np.zeros(3), flipped, np.zeros(3), 0.))
    assert command[0] == pytest.approx(0.44)
    assert abs(command[1]) == pytest.approx(controller.gains.max_body_rate_rad_s)


def test_detector_latches_on_thrust_residual_above_threshold():
    detector = ImpactDetector(1.0)
    assert not detector.update(0., [0., 0., 9.81], 0.15 * 9.81, 0.15)
    assert detector.update(0.001, [12., 0., 9.81], 0.15 * 9.81, 0.15)
    assert detector.update(0.002, [0., 0., 9.81], 0.15 * 9.81, 0.15) and detector.detected_at == 0.001


def test_seeded_plant_applies_draw_and_default_stays_nominal():
    nominal = plant()
    assert all(nominal.physical["draw"][k] == 1. for k in ("mass_scale", "thrust_scale", "motor_tau_scale"))
    drawn = plant(physical_seed=7)
    uid = drawn.model.body("uav").id
    assert drawn.model.body_mass[uid] == pytest.approx(
        nominal.model.body_mass[uid] * drawn.physical["draw"]["mass_scale"])
    assert drawn.physical["target"]["friction"] != nominal.physical["target"]["friction"]


def test_runner_pairs_conditions_and_refuses_existing_output(tmp_path):
    from rm27_sim.experiments.impact_recovery import run
    cfg = json.loads((CONFIGS / "impact_recovery/s1_immediate_stabilization.json").read_text())
    cfg.update(trials_per_cell=2, grid={"attitude_mode": ["normal"]})
    path = tmp_path / "s1.json"
    path.write_text(json.dumps(cfg))
    summary = run(path, tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out/trials.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * len(cfg["conditions"])
    for repeat in range(2):
        cases = {json.dumps(r["case"]) + str(r["physical_seed"]) for r in rows if r["repeat"] == repeat}
        assert len(cases) == 1
    assert set(summary["overall"]) == set(cfg["conditions"])
    with pytest.raises(FileExistsError):
        run(path, tmp_path / "out")


def test_detection_consumes_force_that_actually_integrated_velocity(monkeypatch):
    import mujoco
    from rm27_contact_physics import impact
    p = plant()
    expected = []
    step = mujoco.mj_step
    detect = ImpactDetector.update

    def measured_step(model, data):
        velocity = data.qvel[:3].copy()
        rotation = data.xmat[model.body("uav").id].reshape(3, 3).copy()
        step(model, data)
        expected.append(rotation.T @ ((data.qvel[:3] - velocity) / model.opt.timestep + [0., 0., 9.81]))

    def check_detect(self, timestamp, force, thrust, mass):
        assert force == pytest.approx(expected.pop(0), abs=1e-8)
        return detect(self, timestamp, force, thrust, mass)

    monkeypatch.setattr(impact.mujoco, "mj_step", measured_step)
    monkeypatch.setattr(ImpactDetector, "update", check_detect)
    result = run_trial(p, ImpactCase())
    assert result["detected"] and not result["false_alarm"] and not expected


def test_physical_sweep_does_not_reveal_dynamics_to_controller():
    p = plant()
    values = dict(thrust_to_weight=2., inertia_scale=.75, yaw_torque_per_thrust_m=.009)
    q = plant(physical_values=values)
    assert q.inertia == pytest.approx(p.inertia)
    assert q.mixer == pytest.approx(p.mixer)
    assert q.maximum == p.maximum and q.hover_force == p.hover_force
    command = np.array([p.hover_force, 1., .5, .1])
    assert q.apply_ctbr(command)[0] == pytest.approx(p.apply_ctbr(command)[0])
    capability = q.actuator_capability()
    assert capability['reachable_thrust_to_weight'] == pytest.approx(2.)
    assert capability['physical_inertia_kg_m2'] == pytest.approx(.75 * p.inertia)
    assert capability['physical_yaw_torque_per_thrust_m'] == pytest.approx(.009)
    # PX4 maps normalized output through the fixed hover-throttle calibration.
    per_output = p.hover_force / 4 / RATE_PARAMS['hover_throttle']
    assert q.actuator_capability(per_output)['reachable_thrust_to_weight'] == pytest.approx(
        (2 / 3) / RATE_PARAMS['hover_throttle'])
    nominal = plant(physical_values=dict(thrust_to_weight=3., inertia_scale=1., yaw_torque_per_thrust_m=.018))
    assert run_trial(nominal, ImpactCase()) == run_trial(p, ImpactCase())
    with pytest.raises(ValueError):
        plant(physical_values={'inertia_scale': 0})


def test_parallel_runner_preserves_case_order_and_results(tmp_path):
    from rm27_sim.experiments.impact_recovery import run
    cfg = json.loads((CONFIGS / 'impact_recovery/s1_immediate_stabilization.json').read_text())
    cfg.update(trials_per_cell=2, grid={'normal_speed_m_s': [2.25]},
               conditions={'recover': {'mode': 'stabilize'}})
    config = tmp_path / 'study.json'
    config.write_text(json.dumps(cfg))
    run(config, tmp_path / 'serial')
    run(config, tmp_path / 'parallel', workers=2)
    assert (tmp_path / 'serial/trials.jsonl').read_bytes() == (tmp_path / 'parallel/trials.jsonl').read_bytes()


def test_dynamics_sweep_pairs_cases_and_covers_unique_factor_combinations(tmp_path):
    from rm27_sim.experiments.dynamics_sensitivity import paired_comparison, sweep_points, run
    cfg = json.loads((CONFIGS / 'impact_recovery/dynamics_sensitivity.json').read_text())
    points = sweep_points(cfg)
    assert points[0]['name'] == 'nominal' and len(points) == 14
    assert sum(p['stage'] == 'single_factor' for p in points) == 8
    assert len({tuple(sorted(p['physical_values'].items())) for p in points}) == 14
    a = [dict(cell_index=0, repeat=i, case={'x': i}, physical_seed=i, outcome=outcome)
         for i, outcome in enumerate(['RECOVERED', 'NOT_SETTLED', 'NOT_SETTLED'])]
    b = [dict(r, outcome='RECOVERED') for r in a]
    comparison = paired_comparison(a, b)
    assert comparison['gained'] == 2 and comparison['lost'] == 0
    assert comparison['rate_difference'] == pytest.approx(2/3)
    with pytest.raises(ValueError):
        paired_comparison(a, [dict(r, physical_seed=999) for r in b])
    base = json.loads((CONFIGS / 'impact_recovery/s3/baseline.json').read_text())
    base.update(trials_per_cell=1, grid={'normal_speed_m_s': [2.25]})
    base_path = tmp_path / 'base.json'
    base_path.write_text(json.dumps(base))
    cfg.update(base_study=str(base_path), scan={k: [v] for k, v in cfg['nominal'].items()},
               contacts=cfg['contacts'])
    cfg['free_flight'].update(duration_s=.8, step_start_s=.2, step_end_s=.35, tail_s=.2)
    path = tmp_path / 'sweep.json'
    path.write_text(json.dumps(cfg))
    result = run(path, tmp_path / 'out')
    assert result['passed'] and result['impact_trials'] == 4
    assert len(result['free_flight']) == 8
    summary = result['results']['nominal/soft']
    assert summary['versus_nominal']['proxy']['rate_difference'] == 0.
    assert summary['actuator_capability']['proxy']['controller_inertia_kg_m2'] == pytest.approx(
        summary['actuator_capability']['px4']['controller_inertia_kg_m2'])
    assert result['contact_comparisons']['nominal']['stiff_minus_soft']['proxy']['trials'] == 1


RATE_PARAMS = json.loads((CONFIGS / "impact_recovery/px4_rate_1018.json").read_text())


def test_px4_allocator_normalization_and_airmode_off_desaturation():
    from rm27_contact_physics.px4_rate import Px4Allocator
    allocator = Px4Allocator(plant().mixer)
    out, flags = allocator.allocate(0.4, [0., 0., 0.])
    assert out == pytest.approx([0.4] * 4) and not flags.any()
    out, _ = allocator.allocate(0.4, [0.2, 0., 0.])
    assert np.sort(np.abs(out - 0.4)) == pytest.approx([0.2 / np.sqrt(2)] * 4)
    out, flags = allocator.allocate(0.95, [0.2, 0., 0.])  # top: collective drops, torque kept
    assert out.max() == pytest.approx(1.) and np.ptp(out) == pytest.approx(0.4 / np.sqrt(2))
    assert not flags[:, 0].any()
    out, flags = allocator.allocate(0.05, [0.5, 0., 0.])  # bottom: roll shrinks and is flagged
    assert out.min() == pytest.approx(0.) and flags[0, 0] and not flags[1, 0]


def test_px4_rate_loop_tracks_roll_rate_step_in_free_flight():
    import mujoco
    from rm27_contact_physics.px4_rate import Px4Allocator, Px4RateController
    p = plant()
    p.reset(position=np.array([0., 0., 3.]), rotation=np.eye(3), velocity=np.zeros(3), omega_body=np.zeros(3))
    rate_loop, allocator = Px4RateController(RATE_PARAMS), Px4Allocator(p.mixer)
    for _ in range(200):
        torque = rate_loop.update([3., 0., 0.], p.state.omega_body)
        outputs, rate_loop.saturated = allocator.allocate(p.hover_force / (4 * p.maximum), torque)
        p.data.ctrl[:] = outputs * p.maximum
        mujoco.mj_step(p.model, p.data)
        mujoco.mj_forward(p.model, p.data)
    assert p.state.omega_body == pytest.approx([3., 0., 0.], abs=0.15)


def test_px4_inner_loop_trial_requires_parameters_and_recovers_nominal_case():
    p = plant()
    with pytest.raises(ValueError):
        run_trial(p, ImpactCase(), RecoverySpec(inner_loop="px4"))
    assert run_trial(p, ImpactCase(), RecoverySpec(inner_loop="px4"), rate_params=RATE_PARAMS)["outcome"] == "RECOVERED"


DRONE_V2_MJCF = resolve_asset("components/rm_mjcf/rm27_drone_mjcf/dynamic/model.xml")


@pytest.mark.skipif(not DRONE_V2_MJCF.exists(), reason="rm_mjcf runtime aircraft asset is not configured")
def test_imported_v2_aircraft_matches_profile_and_places_bottom_site_at_gap():
    base = json.loads((CONFIGS / "impact_recovery/contact_base_v2.json").read_text())
    sources = dict(vehicle_config=CONFIGS / "drone_v2.json",
                   dynamics_profile=CONFIGS / "impact_recovery/dynamics_v2_150g.json",
                   target_geometry=CONFIGS / "base_target.json", vehicle_mjcf=DRONE_V2_MJCF)
    p = RM27Plant(base, sources)  # verify_compiled rejects any mass/inertia/actuator mismatch
    uid = p.model.body("uav").id
    assert p.model.nmeshvert == 0 and p.info["rotor_arm_m"] == pytest.approx(0.0275)
    assert sum(p.model.geom_bodyid == uid) == 247  # 246 CAD proxies + the bottom-centre ball
    bumper = p.model.geom("bumper")
    assert bumper.pos[2] - bumper.size[0] < -0.0225 - 0.03  # protrudes below the feet
    n = np.asarray(base["target"]["normal"]) / np.linalg.norm(base["target"]["normal"])
    position, rotation, _, _ = initial_state(p, ImpactCase(gap_m=0.02))
    surface = position + rotation @ bumper.pos - bumper.size[0] * n
    assert (surface - base["target"]["center_m"]) @ n == pytest.approx(0.02, abs=1e-12)
    result = run_trial(p, ImpactCase(), RecoverySpec(), rate_params=RATE_PARAMS)
    assert result["outcome"] == "RECOVERED" and result["bottom_only"]


def test_base_target_is_the_measured_rmuc2027_blue_plate():
    target = json.loads((CONFIGS / "base_target.json").read_text())
    assert target["target_team"] == "blue" and "rmuc2027" in target["field_path"]
    assert target["armor_surface_center_m"][0] > 11. and target["armor_normal"][0] < 0.  # +X base faces the centre
    assert np.degrees(np.arccos(target["armor_normal"][2] / np.linalg.norm(target["armor_normal"]))) == pytest.approx(27.5, abs=0.1)
    assert json.loads((CONFIGS / "base_target_rmuc2026.json").read_text())["armor_surface_center_m"][0] < 0.


def test_setpoint_rules_follow_patnaik_and_rebound_definitions():
    p = plant()
    n = np.asarray(p.physical["target"]["normal"]) / np.linalg.norm(p.physical["target"]["normal"])
    fixed = run_trial(p, ImpactCase(), RecoverySpec(setpoint_backoff_m=0.3))
    patnaik = run_trial(p, ImpactCase(), RecoverySpec(setpoint_rule="patnaik", backoff_gain=0.5))
    # Takeover happens at the first contact sample, a few mm from the impact position.
    assert np.asarray(fixed["setpoint_offset_m"]) @ n == pytest.approx(0.3, abs=0.01)
    assert patnaik["approach_speed_m_s"] == pytest.approx(2.25, abs=0.02)
    assert np.asarray(patnaik["setpoint_offset_m"]) @ n == pytest.approx(0.5 * patnaik["approach_speed_m_s"], abs=0.01)
    rebound = run_trial(p, ImpactCase(), RecoverySpec(setpoint_rule="rebound", backoff_gain=0.2))
    assert rebound["setpoint_offset_m"] != fixed["setpoint_offset_m"]
    with pytest.raises(ValueError):
        run_trial(p, ImpactCase(), RecoverySpec(setpoint_rule="unknown"))


IMU_PARAMS = json.loads((CONFIGS / "impact_recovery/imu_bmi088_px4.json").read_text())


def test_imu_clips_at_full_scale_samples_at_odr_and_px4_filter_lags():
    from rm27_contact_physics.imu import Bmi088Px4
    params = json.loads(json.dumps(IMU_PARAMS))
    params["accel"].update(noise_density_ug_rthz=0., vibration_std_m_s2=0.)
    imu = Bmi088Px4(params, seed=0)
    samples, t = 0, 0.
    for step in range(2000):  # 1 s at 0.5 ms; a 40 g, 40 ms pulse from 0.5 s
        t = step * 5e-4
        force = np.array([0., 0., 9.81 + (40 * 9.81 if 0.5 <= t < 0.54 else 0.)])
        imu.update(t, force, np.zeros(3))
        samples += imu.accel.new_sample
        if t == pytest.approx(0.53):
            assert imu.accel.raw[2] == pytest.approx(24 * 9.81)  # register saturation
        if t == pytest.approx(0.505):
            assert imu.accel.filtered[2] < 0.5 * imu.accel.raw[2]  # 30 Hz PX4 low-pass lags the edge
    assert samples == pytest.approx(1600, abs=2)
    assert imu.accel.clipped_samples == pytest.approx(0.04 * 1600, abs=2)


def test_imu_detection_is_later_with_px4_filter_than_raw():
    p = plant()
    raw = run_trial(p, ImpactCase(), RecoverySpec(detection_source="imu_raw"), imu_params=IMU_PARAMS, imu_seed=1)
    filtered = run_trial(p, ImpactCase(), RecoverySpec(detection_source="imu_filtered"), imu_params=IMU_PARAMS, imu_seed=1)
    assert raw["detected"] and filtered["detected"] and not filtered["false_alarm"]
    assert 0 <= raw["detection_delay_s"] < filtered["detection_delay_s"] < 0.03
    with pytest.raises(ValueError):
        run_trial(p, ImpactCase(), RecoverySpec(detection_source="imu_raw"))


def test_retreat_pushes_away_along_the_normal_before_holding():
    p = plant()
    n = np.asarray(p.physical["target"]["normal"]) / np.linalg.norm(p.physical["target"]["normal"])
    calm = run_trial(p, ImpactCase(), RecoverySpec(), trace=True)["trace"]
    kick = run_trial(p, ImpactCase(), RecoverySpec(retreat="normal", retreat_s=0.1, retreat_accel_m_s2=15.), trace=True)["trace"]
    contact = np.argmax(kick[:, -1] > 0)
    away = lambda tr: (tr[contact + 100, 1:4] - tr[contact, 1:4]) @ n  # 0.1 s after the impact
    assert away(kick) > away(calm) + 0.02
    with pytest.raises(ValueError):
        run_trial(p, ImpactCase(), RecoverySpec(retreat="sideways"))
