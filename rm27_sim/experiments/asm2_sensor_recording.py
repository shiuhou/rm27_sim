"""Synthetic sensor-generation boundary for frozen physical trajectory replay.

Only this generator sees native truth. Sensor packets never contain truth state,
contact labels, initial true attitude, controller references or motor telemetry.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from rm27_contact_physics.imu import Bmi088Px4
from rm27_contact_physics.recovery import GRAVITY, ImpactDetector
from rm27_sim.simulation import asm2 as native

SOURCE = 'TRUTH_GENERATED_SYNTHETIC'


def packet(kind, acquired, delivered, value, *, clipped=False):
    """Materialize one synthetic measurement, with no native state namespace."""
    return dict(kind=kind, acquisition_s=float(acquired), delivery_s=float(delivered),
                value=value, source=SOURCE, valid=True, clipped=bool(clipped))


class SensorRecorder:
    """COM-mounted synthetic instruments; inputs stay inside generator boundary."""
    def __init__(self, config, seed, *, start_s, native_dt_s, held_thrust_N, mass_kg):
        self.config = config
        self.start = float(start_s)
        self.dt = float(native_dt_s)
        if not np.isfinite([self.start, self.dt, held_thrust_N, mass_kg]).all() or self.dt <= 0 or mass_kg <= 0:
            raise ValueError('Invalid instrument clock or plant constants')
        streams = np.random.SeedSequence(seed).spawn(3)
        self.imu = Bmi088Px4(config['imu'], streams[0])
        self.bias = np.random.default_rng(streams[1]).normal(0., config['gyro_bias_std_rad_s'], 3)
        self.external_rng = np.random.default_rng(streams[2])
        self.held, self.mass = float(held_thrust_N), float(mass_kg)
        self.detector = ImpactDetector(config['detector_threshold_g'], config['detector_persistence'])
        self.packets = []
        self.last_tick = -1
        self.external_index = 0
        self.impact_emitted = False

    def observe(self, tick, position, velocity, rotation, omega, *, before_velocity=None, before_rotation=None):
        """Observe consecutive native endpoints; tick0 primes declared held thrust."""
        if tick != self.last_tick + 1:
            raise ValueError('Expected consecutive integer native ticks')
        self.last_tick = tick
        now = self.start + tick * self.dt
        if tick == 0:
            force = np.array([0., 0., self.held / self.mass])
        else:
            if before_velocity is None or before_rotation is None:
                raise ValueError('Native interval derivative requires preceding state')
            force = np.asarray(before_rotation).T @ ((np.asarray(velocity)-before_velocity)/self.dt + [0., 0., GRAVITY])
        old_counts = (self.imu.accel.clipped_samples, self.imu.gyro.clipped_samples)
        self.imu.update(now, force, np.asarray(omega)+self.bias)
        for kind, channel, old in zip(('accel', 'gyro'), (self.imu.accel, self.imu.gyro), old_counts):
            if channel.new_sample:
                clipped = channel.clipped_samples > old or np.any(np.abs(channel.raw) >= channel.full_scale)
                self.packets.append(packet(kind, now, now, channel.filtered.tolist(), clipped=clipped))
        if self.imu.accel.new_sample:
            self.detector.update(now, self.imu.accel.filtered, self.held, self.mass)
            if self.detector.detected_at is not None and not self.impact_emitted:
                self.impact_emitted = True
                self.packets.append(packet('impact', now, now+self.config['event_latency_s'],
                    dict(lower_s=self.start, upper_s=now)))
        period = self.config['translation_period_s']
        # Integer native tick defines actual capture; do not backdate to deadline.
        if tick*self.dt + 1e-12 >= self.external_index*period:
            self.external_index += 1
            p = np.asarray(position)+self.external_rng.normal(0., self.config['position_noise_m'], 3)
            v = np.asarray(velocity)+self.external_rng.normal(0., self.config['velocity_noise_m_s'], 3)
            self.packets.append(packet('external', now, now+self.config['translation_latency_s'],
                dict(position_m=p.tolist(), velocity_m_s=v.tolist())))


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _write_lines(path, rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())


def record_archived_branch(branch, output, *, sensor_seed, duration_s=.35):
    """Replay held archived controls; verify each completed saved endpoint exactly.

    Returns metadata and creates sensors/truth/actuators files separately. It runs
    native replay only; no estimator or recovery policy is executed.
    """
    if not np.isfinite(duration_s) or not 0 < duration_s <= .35:
        raise ValueError('Registered replay window must be within0.35s')
    branch, output = Path(branch), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    summary_path, trajectory_path = branch/'summary.json', branch/'trajectory.npz'
    summary = json.loads(summary_path.read_text())
    if native.digest(trajectory_path) != summary['trajectory_sha256']:
        raise ValueError('Archived trajectory hash mismatch')
    model_path = branch/'model.mjb'
    if not model_path.exists():
        model_path = branch.parent/'scene.mjb'
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        states = archive['integration'].copy()
    if np.any(np.diff(states[:, 0]) <= 0):
        raise ValueError('Saved physical times must increase')
    dt, start = float(model.opt.timestep), float(states[0, 0])
    joint = model.joint('asm2_free').id
    qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    body = model.body('uav').id
    if np.any(model.body_ipos[body] != 0):
        raise ValueError('COM synthetic instrument requires body origin at COM')
    motors = np.array([model.actuator(f'motor_{i}').id for i in range(4)])
    activations = model.actuator_actadr[motors]
    spec_path = native.ROOT/'configs/asm2_recovery_validation.json'
    spec = json.loads(spec_path.read_text())
    imu_path = native.ROOT/spec['imu_config']
    config = {**spec['sensors'], 'imu': json.loads(imu_path.read_text())}
    if (config['detector_threshold_g'], config['detector_persistence']) != (1., 1):
        raise ValueError('Frozen production detector required')
    mass = float(model.body_mass[body])
    case = summary['case']
    held = float(case.get('held_thrust_N', mass*GRAVITY*case.get('hover_fraction', 1.)))
    recorder = SensorRecorder(config, sensor_seed, start_s=start, native_dt_s=dt, held_thrust_N=held, mass_kg=mass)
    mujoco.mj_setState(model, data, states[0], native.STATE)
    mujoco.mj_forward(model, data)
    truth, actuators = [], []
    snapshot = np.empty(mujoco.mj_stateSize(model, native.STATE))

    def state():
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[qadr+3:qadr+7])
        return (data.qpos[qadr:qadr+3].copy(), data.qvel[dadr:dadr+3].copy(),
                rotation.reshape(3, 3), data.qvel[dadr+3:dadr+6].copy())

    def save(tick, current):
        p, v, r, w = current
        timestamp = float(data.time)
        truth.append(dict(time_s=timestamp, native_tick=tick, position=p.tolist(), velocity=v.tolist(),
                          rotation=r.tolist(), omega=w.tolist(), source='OFFLINE_TRUTH_ONLY'))
        actuators.append(dict(time_s=timestamp, native_tick=tick, command_N=data.ctrl[motors].tolist(),
                              activation_N=data.act[activations].tolist(), interval_actuator_force_N=data.actuator_force[motors].tolist()))

    current = state()
    recorder.observe(0, *current)
    save(0, current)
    tick = boundaries = 0
    max_tick = int(np.floor(duration_s/dt + 1e-9))
    for index in range(len(states)-1):
        # Following snapshot stores the held input over this preceding bracket.
        mujoco.mj_setState(model, scratch, states[index+1], native.STATE)
        data.ctrl[:] = scratch.ctrl
        ticks = int(round((states[index+1, 0]-states[index, 0])/dt))
        if abs(ticks*dt-(states[index+1, 0]-states[index, 0])) > 1e-9:
            raise ValueError('Archive bracket does not span integer native ticks')
        if tick+ticks > max_tick:
            break  # Only complete verified archive brackets; no unverified tail.
        for _ in range(ticks):
            before = current
            mujoco.mj_step(model, data)
            tick += 1
            current = state()
            recorder.observe(tick, *current, before_velocity=before[1], before_rotation=before[2])
            save(tick, current)
        mujoco.mj_getState(model, data, snapshot, native.STATE)
        if snapshot.tobytes() != states[index+1].tobytes():
            raise AssertionError(f'Archived endpoint differs at bracket{index}')
        boundaries += 1
    _write_lines(output/'sensors.jsonl.gz', sorted(recorder.packets, key=lambda row: (row['delivery_s'], row['acquisition_s'], row['kind'])))
    _write_lines(output/'truth.jsonl.gz', truth)
    _write_lines(output/'actuators.jsonl.gz', actuators)
    contacts_path = branch/'contacts.json'
    # Label source link only; do not expose solver contacts in sensor packets.
    contact_labels = dict(source_path=str(contacts_path), sha256=native.digest(contacts_path),
                          first_contact=summary['first_contact'], first_forbidden=summary['first_forbidden'])
    _write_json(output/'contact_labels_offline.json', contact_labels)
    paths = [Path(__file__), summary_path, trajectory_path, contacts_path, model_path, spec_path, imu_path,
             native.ROOT/'packages/physics/src/rm27_contact_physics/imu.py',
             native.ROOT/'packages/physics/src/rm27_contact_physics/recovery.py']
    metadata = dict(kind='SYNTHETIC_SENSOR_GENERATION_FROM_VERIFIED_ARCHIVED_REPLAY',
                    branch=str(branch), physical_seed=case['seed'], sensor_seed=sensor_seed,
                    sensor_source=SOURCE, native_dt_s=dt, requested_duration_s=duration_s,
                    recorded_duration_s=tick*dt, native_replay_steps=tick, verified_endpoints=boundaries,
                    sensor_packets=len(recorder.packets), imu_lever_arm_body_m=[0., 0., 0.],
                    sensor_packet_schema='ONLY_KIND_ACQUISITION_DELIVERY_VALUE_SOURCE_VALID_CLIPPED',
                    imu_filter_stage='PX4_FILTERED', accel_initial_sample='PRIMED_FROM_DECLARED_HELD_THRUST_NOT_MEASURED_DERIVATIVE',
                    imu_group_delay='INTERNAL_FILTER_RESPONSE; ACQUISITION_TIMESTAMP_NOT_SHIFTED',
                    initial_attitude_truth_packet=False, external_source='SYNTHETIC_EXTERNAL_POSITION_VELOCITY_NOT_VISION',
                    frame_convention='BODY_FLU; WORLD_ENU', acquisition='INTEGER_NATIVE_TICKS_ACTUAL_ENDPOINT',
                    synthetic_imu_latency_s=0., event_latency_s=config['event_latency_s'],
                    contact_event_interval='STREAM_START_TO_SENSOR_DETECTION; NOT_NATIVE_CONTACT_TIME',
                    physics_and_controls='Frozen archived controls; no new policy/controller/trajectory search',
                    mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                    config=config, source_sha256={str(p): native.digest(p) for p in paths},
                    output_sha256={p.name: native.digest(p) for p in output.iterdir() if p.is_file()},
                    limitations=['Synthetic COM-mounted IMU; physical mounting offset unregistered.',
                                 'No initial true-attitude packet; estimator must establish its own initialization.',
                                 'Body force uses preceding native orientation and interval COM velocity difference.',
                                 'Zero IMU transport latency is declared synthetic assumption; external/event latency fixed models.',
                                 'Recorded prior physical cases are retrospective, not prospective unseen evidence.'],
                    wall_s=perf_counter()-started)
    _write_json(output/'metadata.json', metadata)
    return metadata
