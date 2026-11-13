"""Synthetic sensing for the standalone low-energy recovery fixture.

Native states drive simulated instruments. The estimate path uses a single noisy
attitude alignment, filtered gyro, and an optional delayed external position and
velocity stream. Contact timing comes from the filtered accelerometer residual.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from rm27_contact_physics.imu import Bmi088Px4
from rm27_contact_physics.model import VehicleEstimate
from rm27_contact_physics.recovery import GRAVITY, ImpactDetector


EPS = 1e-9
GRAVITY_UP = np.array([0., 0., GRAVITY])


def _outage(case, key):
    value = case.get(key)
    if value is None:
        return None
    start, end = map(float, value)
    if not np.isfinite([start, end]).all() or not 0. <= start < end:
        raise ValueError(f'{key} requires a finite increasing interval')
    return start, end


def _offline(window, timestamp):
    return window is not None and window[0]-EPS <= timestamp < window[1]-EPS


class RecoverySensors:
    """Keep contact, attitude, and translation clocks independent."""

    def __init__(self, config, case, seed, initial_state):
        mode = case.get('sensor_mode', 'aided')
        if mode not in ('aided', 'imu_only'):
            raise ValueError('Unknown recovery sensor mode')
        self.aided = mode == 'aided'
        imu_seed, alignment_seed, translation_seed = np.random.SeedSequence(seed).spawn(3)
        self.imu = Bmi088Px4(config['imu'], imu_seed)
        alignment_rng = np.random.default_rng(alignment_seed)
        self._translation_rng = np.random.default_rng(translation_seed)
        self._mass_kg = float(config.get('mass_kg', .142))
        self._thrust_estimate_N = float(case.get(
            'held_thrust_N', self._mass_kg * GRAVITY * case.get('hover_fraction', 1.)))
        self._event_latency_s = float(case.get('event_latency_s', config.get('event_latency_s', .0025)))
        self._gyro_max_age_s = float(config.get('gyro_max_age_s', .010))
        self._translation_period_s = float(config.get('translation_period_s', .020))
        self._translation_latency_s = float(config.get('translation_latency_s', .015))
        self._translation_max_age_s = float(config.get('translation_max_age_s', .050))
        self._position_noise_m = float(config.get('position_noise_m', .002))
        self._velocity_noise_m_s = float(config.get('velocity_noise_m_s', .012))
        alignment_std = float(config.get('attitude_initial_noise_deg', .35))
        gyro_bias_std = float(config.get('gyro_bias_std_rad_s', .004))
        limits = [self._mass_kg, self._gyro_max_age_s, self._translation_period_s,
                  self._translation_max_age_s]
        nonnegative = [self._thrust_estimate_N, self._event_latency_s, self._translation_latency_s,
                       self._position_noise_m, self._velocity_noise_m_s, alignment_std, gyro_bias_std]
        if not np.isfinite(limits + nonnegative).all() or min(limits) <= 0 or min(nonnegative) < 0:
            raise ValueError('Invalid recovery sensor configuration')
        self.detector = ImpactDetector(config.get('detector_threshold_g', 1.),
                                       config.get('detector_persistence', 1))
        self._gyro_outage = _outage(case, 'gyro_outage_s')
        self._translation_outage = _outage(case, 'translation_outage_s')

        alignment_error = np.deg2rad(alignment_rng.normal(0., alignment_std, 3))
        self._rotation = initial_state.rotation @ Rotation.from_rotvec(alignment_error).as_matrix()
        self._gyro_bias = alignment_rng.normal(0., gyro_bias_std, 3)
        self._gyro_time = self._gyro = self._attitude_fault = None
        self._latest_translation = None
        self._translation_queue = deque()
        self._translation_index = 0
        self._start = self._last_endpoint = float(initial_state.timestamp)
        self._peak_residual_g = 0.
        self._accelerometer_samples = 0
        # The first derivative sample is unavailable at release. Prime the filters
        # with the declared held thrust and sample the initial gyro normally.
        self._sample_imu(self._start, [0., 0., self._thrust_estimate_N / self._mass_kg],
                         initial_state.omega_body)
        self._capture_translation(initial_state)

    def _sample_imu(self, timestamp, specific_force, omega_body):
        self.imu.update(timestamp, specific_force, omega_body + self._gyro_bias)
        if self.imu.accel.new_sample:
            signal = self.imu.accel.filtered
            residual_g = np.linalg.norm(
                signal - [0., 0., self._thrust_estimate_N / self._mass_kg]) / GRAVITY
            self._accelerometer_samples += 1
            if np.isfinite(residual_g):
                self._peak_residual_g = max(self._peak_residual_g, float(residual_g))
            self.detector.update(timestamp, signal, self._thrust_estimate_N, self._mass_kg)
        if not self.imu.gyro.new_sample or _offline(self._gyro_outage, timestamp):
            return
        previous = self._start if self._gyro_time is None else self._gyro_time
        gyro = self.imu.gyro.filtered
        if not np.isfinite(gyro).all() or not np.isfinite(self.imu.gyro.raw).all():
            self._attitude_fault = 'GYRO_NONFINITE'
        elif np.any(np.abs(self.imu.gyro.raw) >= self.imu.gyro.full_scale):
            self._attitude_fault = 'GYRO_CLIPPED'
        elif timestamp - previous > self._gyro_max_age_s + EPS:
            self._attitude_fault = 'GYRO_GAP_LOST_ALIGNMENT'
        if self._attitude_fault is None:
            mean_gyro = gyro if self._gyro is None else .5 * (gyro + self._gyro)
            increment = Rotation.from_rotvec(mean_gyro * (timestamp - previous)).as_matrix()
            self._rotation = self._rotation @ increment
        self._gyro, self._gyro_time = gyro.copy(), timestamp

    def _capture_translation(self, state):
        timestamp = state.timestamp
        if not self.aided or timestamp + EPS < self._start + self._translation_index * self._translation_period_s:
            return
        # Take one measurement at the actual physics endpoint, avoiding backdated
        # captures if a sample deadline falls between two endpoints.
        self._translation_index = int(np.floor(
            (timestamp - self._start + EPS) / self._translation_period_s)) + 1
        if _offline(self._translation_outage, timestamp):
            return
        p = state.position + self._translation_rng.normal(0., self._position_noise_m, 3)
        v = state.velocity + self._translation_rng.normal(0., self._velocity_noise_m_s, 3)
        self._translation_queue.append((timestamp + self._translation_latency_s, timestamp, p, v))

    def update(self, before, after):
        """Observe one completed native interval without touching the plant."""
        dt = after.timestamp - before.timestamp
        if not np.isfinite(dt) or dt <= 0 or abs(before.timestamp - self._last_endpoint) > EPS:
            raise ValueError('Expected consecutive native physics intervals')
        self._last_endpoint = after.timestamp
        accel_world = (after.velocity - before.velocity) / dt
        specific_force = before.rotation.T @ (accel_world + GRAVITY_UP)
        self._sample_imu(after.timestamp, specific_force, after.omega_body)
        self._capture_translation(after)

    def detected_at(self, now):
        acquired = self.detector.detected_at
        if acquired is not None and acquired + self._event_latency_s <= now + EPS:
            return acquired
        return None

    def detector_status(self):
        acquired = self.detector.detected_at
        return dict(
            acquired_s=acquired,
            delivery_s=None if acquired is None else acquired + self._event_latency_s,
            peak_residual_g=self._peak_residual_g,
            accelerometer_samples=self._accelerometer_samples,
            accelerometer_clipped_samples=self.imu.accel.clipped_samples,
            gyro_input_clipped_samples=self.imu.gyro.clipped_samples,
        )

    def estimate(self, now):
        """Return a control estimate only while both constituent streams are valid."""
        while self._translation_queue and self._translation_queue[0][0] <= now + EPS:
            self._latest_translation = self._translation_queue.popleft()
        gyro_age = None if self._gyro_time is None else now - self._gyro_time
        attitude_reason = self._attitude_fault
        if attitude_reason is None:
            if gyro_age is None:
                attitude_reason = 'GYRO_UNAVAILABLE'
            elif not -EPS <= gyro_age <= self._gyro_max_age_s + EPS:
                attitude_reason = 'GYRO_STALE_OR_FUTURE'
        rotation = omega = None
        if attitude_reason is None:
            omega = self._gyro
            rotation = self._rotation @ Rotation.from_rotvec(omega * max(0., gyro_age)).as_matrix()

        translation_time = translation_age = p = v = None
        translation_reason = None
        if not self.aided:
            translation_reason = 'NO_EXTERNAL_TRANSLATION'
        elif self._latest_translation is None:
            translation_reason = 'TRANSLATION_PENDING'
        else:
            _, translation_time, p, v = self._latest_translation
            translation_age = now - translation_time
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                translation_reason = 'TRANSLATION_NONFINITE'
            elif not -EPS <= translation_age <= self._translation_max_age_s + EPS:
                translation_reason = 'TRANSLATION_STALE_OR_FUTURE'

        status = dict(
            attitude_valid=attitude_reason is None, translation_valid=translation_reason is None,
            attitude_timestamp_s=self._gyro_time, translation_timestamp_s=translation_time,
            attitude_age_s=gyro_age, translation_age_s=translation_age,
            attitude_reason=attitude_reason, translation_reason=translation_reason,
            rotation=None if rotation is None else rotation.tolist(),
            omega_body_rad_s=None if omega is None else omega.tolist(),
        )
        if not status['attitude_valid'] or not status['translation_valid']:
            return None, status
        position = p + max(0., translation_age) * v
        return VehicleEstimate(position, v.copy(), rotation, omega.copy(), now,
                               'SYNTHETIC_GYRO_AND_DELAYED_EXTERNAL_TRANSLATION'), status
