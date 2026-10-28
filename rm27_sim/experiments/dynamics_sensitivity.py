"""Paired physical dynamics sensitivity, with free-flight checks before impacts."""
import copy
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import mujoco
import numpy as np
from scipy.stats import binomtest

from .impact_recovery import ROOT, load_study, provenance, run as run_impacts
from rm27_contact_physics.plant import RM27Plant, RateActuation
from rm27_contact_physics.recovery import RecoveryController


def sweep_points(config):
    """Nominal, unique single factors, then previously untested yaw/inertia pairs."""
    nominal = config['nominal']
    points, seen = [], set()

    def add(stage, name, values):
        key = tuple(sorted(values.items()))
        if key not in seen:
            seen.add(key)
            points.append(dict(stage=stage, name=name, physical_values=values))

    add('single_factor', 'nominal', dict(nominal))
    for factor, values in config['scan'].items():
        for value in values:
            add('single_factor', f'{factor}_{value:g}', dict(nominal, **{factor: value}))
    for inertia in config['scan']['inertia_scale']:
        for yaw in config['scan']['yaw_torque_per_thrust_m']:
            add('yaw_inertia', f'inertia_{inertia:g}_yaw_{yaw:g}',
                dict(nominal, inertia_scale=inertia, yaw_torque_per_thrust_m=yaw))
    return points


def free_flight(study, values, inner_loop, settings, axis=None):
    """Hover or a bounded rate pulse in empty air, through the impact actuation path.

    The target plate remains far below. No constraint/teleport hides an unstable
    response; rate checks report translational drift but score rate tracking.
    """
    plant = RM27Plant(study['base'], study['sources_plant'], physical_values=values)
    state = plant.reset(position=np.array([0., 0., 20.]), velocity=np.zeros(3), rotation=np.eye(3),
                        omega_body=np.array([.1, -.1, .1]) if axis is None else np.zeros(3))
    actuation = RateActuation(plant, inner_loop, study['rate_params'])
    hold = RecoveryController(plant.info['mass_kg'], 4 * plant.maximum, state.position.copy())
    dt = float(plant.model.opt.timestep)
    divider = int(round(plant.cfg['control_dt_s'] / dt))
    errors, tail, peak, saturation, touched, finite = [], [], 0., 0., False, True
    command = np.array([plant.hover_force, 0., 0., 0.])
    for step in range(int(round(settings['duration_s'] / dt))):
        t = state.timestamp
        rate = np.zeros(3)
        if axis is not None and settings['step_start_s'] <= t < settings['step_end_s']:
            rate[axis] = settings['step_rate_rad_s']
        if step % divider == 0:
            command = hold.command(state) if axis is None else np.r_[plant.hover_force, rate]
        saturation += dt * actuation.apply(command, update_proxy=step % divider == 0)
        mujoco.mj_step(plant.model, plant.data)
        touched |= plant.data.ncon > 0
        mujoco.mj_forward(plant.model, plant.data)
        state = plant.state
        finite = bool(np.isfinite(plant.data.qpos).all() and np.isfinite(plant.data.qvel).all())
        peak = max(peak, float(np.linalg.norm(state.omega_body)))
        error = float(np.linalg.norm(state.omega_body - rate))
        errors.append(error)
        if state.timestamp >= settings['duration_s'] - settings['tail_s']:
            tilt = float(np.degrees(np.arccos(np.clip(state.rotation[2, 2], -1., 1.))))
            tail.append((error, float(np.linalg.norm(state.position - hold.setpoint)),
                         float(np.linalg.norm(state.velocity)), tilt))
        if not finite or touched or peak > settings['abort_body_rate_rad_s']:
            break
    maxima = np.max(tail, axis=0).tolist() if tail else [None] * 4
    reasons = []
    if not finite:
        reasons.append('NONFINITE')
    if touched:
        reasons.append('CONTACT')
    if not tail:
        reasons.append('NO_SETTLED_WINDOW')
    if peak >= settings['max_peak_body_rate_rad_s']:
        reasons.append('EXCESS_BODY_RATE')
    if tail:
        limits = ([1., settings['max_hover_position_error_m'], settings['max_hover_speed_m_s'],
                   settings['max_hover_tilt_deg']] if axis is None else [settings['max_settled_rate_error_rad_s']])
        names = ['RATE_NOT_SETTLED', 'POSITION_NOT_SETTLED', 'SPEED_NOT_SETTLED', 'TILT_NOT_SETTLED']
        reasons.extend(name for value, limit, name in zip(maxima, limits, names) if value >= limit)
    return dict(check='hover' if axis is None else ('roll', 'pitch', 'yaw')[axis],
                inner_loop=inner_loop, physical_values=values, passed=not reasons, reasons=reasons,
                finite=finite, touched=touched, simulated_s=state.timestamp,
                peak_body_rate_rad_s=peak, rate_error_rms_rad_s=float(np.sqrt(np.mean(np.square(errors)))),
                tail_max_rate_error_rad_s=maxima[0], tail_max_position_error_m=maxima[1],
                tail_max_speed_m_s=maxima[2], tail_max_tilt_deg=maxima[3],
                motor_saturation_time_s=saturation, actuator_capability=plant.actuator_capability(actuation.per_output))


def paired_comparison(reference, candidate):
    """Paired outcome differences, exact McNemar p, deterministic paired-bootstrap CI."""
    def indexed(rows):
        return {(r['cell_index'], r['repeat']): r for r in rows}

    a, b = indexed(reference), indexed(candidate)
    if len(a) != len(reference) or len(b) != len(candidate) or not a or a.keys() != b.keys():
        raise ValueError('Paired comparison requires unique identical case IDs')
    gains = losses = 0
    for key in a:
        x, y = a[key], b[key]
        if x['case'] != y['case'] or x['physical_seed'] != y['physical_seed']:
            raise ValueError('Paired cases or physical seeds differ')
        ok_x, ok_y = x['outcome'] == 'RECOVERED', y['outcome'] == 'RECOVERED'
        gains += not ok_x and ok_y
        losses += ok_x and not ok_y
    n = len(a)
    draws = np.random.default_rng(4096).multinomial(n, np.array([gains, losses, n-gains-losses]) / n, 4000)
    ci = np.quantile((draws[:, 0] - draws[:, 1]) / n, [.025, .975]).tolist()
    return dict(trials=n, gained=int(gains), lost=int(losses), rate_difference=(gains-losses)/n,
                difference_paired_bootstrap95=ci,
                mcnemar_exact_p=1. if gains+losses == 0 else float(binomtest(int(gains), int(gains+losses)).pvalue))


def _check(study, settings, task):
    point, inner, axis = task
    return dict(point=point['name'], **free_flight(study, point['physical_values'], inner, settings, axis))


def run(config_path, output, *, workers=1):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Dynamics sensitivity requires schema_version 1')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    study = load_study(ROOT / config['base_study'])
    points = sweep_points(config)
    output.mkdir(parents=True, exist_ok=False)
    cfg_dir = output / 'configs'
    cfg_dir.mkdir()
    tasks = [(p, inner, axis) for p in points for inner in ('proxy', 'px4') for axis in (None, 0, 1, 2)]
    check = partial(_check, study, config['free_flight'])
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            checks = list(pool.map(check, tasks, chunksize=2))
    else:
        checks = list(map(check, tasks))
    (output / 'free_flight.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f'Free flight: {sum(c["passed"] for c in checks)}/{len(checks)} checks pass; impacts retain all conditions', flush=True)
    results, contact_comparisons = {}, {}
    reference_rows = {}
    for point in points:
        contact_rows = {}
        for contact, contact_values in config['contacts'].items():
            name = f'{point["name"]}/{contact}'
            cfg = copy.deepcopy(study['cfg'])
            cfg.update(study='Physical dynamics sensitivity: '+name, physical_values=point['physical_values'],
                       conditions={inner: dict(mode='stabilize', inner_loop=inner, setpoint_backoff_m=.3)
                                   for inner in ('proxy', 'px4')})
            cfg['target_overrides'] = contact_values
            cfg['uncertainty_overrides'] = dict(cfg.get('uncertainty_overrides', {}),
                friction_range=[contact_values['friction']]*2,
                contact_damping_range=[contact_values['solref_damping_ratio']]*2,
                contact_time_scale_range=[1., 1.])
            config_file = cfg_dir / f'{point["name"]}_{contact}.json'
            config_file.write_text(json.dumps(cfg, indent=2)+'\n')
            summary = run_impacts(config_file, output / name, workers=workers)
            rows = [json.loads(line) for line in (output / name / 'trials.jsonl').read_text().splitlines()]
            by_inner = {inner: [r for r in rows if r['condition'] == inner] for inner in ('proxy', 'px4')}
            contact_rows[contact] = by_inner
            if point['name'] == 'nominal':
                reference_rows[contact] = by_inner
            results[name] = dict(stage=point['stage'], physical_values=point['physical_values'],
                overall=summary['overall'], by_cell=summary['by_cell'],
                versus_nominal={inner: paired_comparison(reference_rows[contact][inner], by_inner[inner])
                                for inner in ('proxy', 'px4')},
                px4_versus_proxy=paired_comparison(by_inner['proxy'], by_inner['px4']),
                actuator_capability={inner: next(c['actuator_capability'] for c in checks
                                                if c['point'] == point['name'] and c['inner_loop'] == inner
                                                and c['check'] == 'hover') for inner in ('proxy', 'px4')},
                drawn_reachable_thrust_to_weight={inner: dict(zip(('min', 'median', 'max'), np.quantile(
                    [r['actuator_capability']['reachable_thrust_to_weight'] for r in by_inner[inner]], [0., .5, 1.]).tolist()))
                    for inner in ('proxy', 'px4')})
            (output / 'summary.json').write_text(json.dumps(dict(passed=False, completed_conditions=len(results),
                expected_conditions=len(points)*len(config['contacts']), config=config, results=results), indent=2)+'\n')
            print(name, {k: round(v['recovery_rate'], 4) for k, v in summary['overall'].items()}, flush=True)
        reference_contact = next(iter(contact_rows))
        contact_comparisons[point['name']] = {
            f'{contact}_minus_{reference_contact}': {
                inner: paired_comparison(contact_rows[reference_contact][inner], rows[inner])
                for inner in ('proxy', 'px4')}
            for contact, rows in contact_rows.items() if contact != reference_contact}
    report = dict(passed=True, scope='headless engineering sensitivity; approximate PX4, not firmware SITL',
                  config=config, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  provenance=provenance(study['path'], study['cfg'], study['sources']),
                  free_flight_settings=config['free_flight'], free_flight=checks, results=results,
                  contact_comparisons=contact_comparisons,
                  impact_trials=sum(v['trials'] for r in results.values() for v in r['overall'].values()),
                  limitations='Physical dynamics, bottom ball and contact are assumptions. Controller state remains truth. '
                              'Failed free-flight checks remain in impact results; paired uncertainty is conditional on this fixed case set.')
    report['limitations'] += (' PX4-versus-proxy includes their different existing normalized-output force ceilings; '
                              'it does not isolate PID algorithm quality. Sensitivity is not parameter identification.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
