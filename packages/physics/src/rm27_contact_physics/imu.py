"""BMI088-like IMU sampling with full-scale clipping and the PX4 accel/gyro filter chain.

Driven once per physics step with truth specific force and body rate. Samples are taken at
each sensor's output data rate from the latest physics state (zero-order hold), clipped at
full scale, passed through the sensor's internal digital low-pass, clipped again (register
saturation) and given white noise plus an optional white "vibration" term. PX4 then applies
its second-order low-pass (IMU_ACCEL_CUTOFF / IMU_GYRO_CUTOFF) at the sensor rate. Values
in the configuration are datasheet/driver assumptions unless stated otherwise.
"""
from __future__ import annotations

import numpy as np

from .px4_rate import LowPass2p
from .recovery import GRAVITY


class _Channel:
    def __init__(self, odr_hz, full_scale, internal_cutoff_hz, px4_cutoff_hz, noise_std, rng):
        self.period = 1. / float(odr_hz)
        self.full_scale = float(full_scale)
        self.internal = LowPass2p(odr_hz, internal_cutoff_hz)
        self.px4 = LowPass2p(odr_hz, px4_cutoff_hz)
        self.noise_std = float(noise_std)
        self.rng = rng
        self.next_time = None
        self.raw = self.filtered = None
        self.clipped_samples = 0
        self.new_sample = False

    def update(self, timestamp, truth):
        self.new_sample = False
        if self.next_time is None:
            self.next_time = timestamp
            self.internal.reset(np.clip(truth, -self.full_scale, self.full_scale))
            self.px4.reset(np.clip(truth, -self.full_scale, self.full_scale))
        while timestamp >= self.next_time - 1e-12:
            clipped = np.clip(truth, -self.full_scale, self.full_scale)
            self.clipped_samples += int(np.any(np.abs(truth) >= self.full_scale))
            raw = self.internal.apply(clipped) + self.rng.normal(0., self.noise_std, 3)
            self.raw = np.clip(raw, -self.full_scale, self.full_scale)
            self.filtered = self.px4.apply(self.raw)
            self.next_time += self.period
            self.new_sample = True


class Bmi088Px4:
    """Accelerometer outputs in m/s^2 (specific force), gyro in rad/s, both body FLU."""

    def __init__(self, params, seed=None):
        rng = np.random.default_rng(seed)
        a, g = params["accel"], params["gyro"]
        accel_noise = np.hypot(a["noise_density_ug_rthz"] * 1e-6 * GRAVITY * np.sqrt(a["noise_bandwidth_hz"]),
                               a.get("vibration_std_m_s2", 0.))
        gyro_noise = np.deg2rad(g["noise_density_dps_rthz"]) * np.sqrt(g["noise_bandwidth_hz"])
        self.accel = _Channel(a["odr_hz"], a["range_g"] * GRAVITY, a["internal_cutoff_hz"],
                              params["px4_accel_cutoff_hz"], accel_noise, rng)
        self.gyro = _Channel(g["odr_hz"], np.deg2rad(g["range_dps"]), g["internal_cutoff_hz"],
                             params["px4_gyro_cutoff_hz"], gyro_noise, rng)

    def update(self, timestamp, specific_force_body, omega_body):
        self.accel.update(timestamp, np.asarray(specific_force_body, float))
        self.gyro.update(timestamp, np.asarray(omega_body, float))
