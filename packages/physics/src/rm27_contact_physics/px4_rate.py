"""Approximate PX4 multicopter rate loop and control allocation (no MuJoCo imports).

Model of MulticopterRateControl (gains K*(P, I, D), integrator with error-dependent
attenuation and limit, D on the filtered gyro derivative, yaw torque low-pass) and of the
pseudo-inverse allocator with PX4 normalization (roll/pitch columns to +/-1/sqrt(2),
yaw to +/-1, collective 1 -> all motors 1) plus airmode-off sequential desaturation.
Written from the published PX4 structure, not verified line by line against v1.17.
"""
from __future__ import annotations

import numpy as np


class LowPass2p:
    """Second-order Butterworth biquad (PX4 LowPassFilter2p form), vector-valued."""

    def __init__(self, sample_hz, cutoff_hz, size=3):
        self.size = size
        if cutoff_hz <= 0 or cutoff_hz >= sample_hz / 2:
            self.b = None
            return
        fr = sample_hz / cutoff_hz
        ohm = np.tan(np.pi / fr)
        c = 1. + 2. * np.cos(np.pi / 4.) * ohm + ohm * ohm
        self.b = np.array([ohm * ohm, 2. * ohm * ohm, ohm * ohm]) / c
        self.a = np.array([2. * (ohm * ohm - 1.) / c, (1. - 2. * np.cos(np.pi / 4.) * ohm + ohm * ohm) / c])
        self.d1 = self.d2 = None

    def reset(self, value):
        value = np.asarray(value, float)
        if self.b is None:
            return
        # Steady-state delay elements for a constant input.
        self.d1 = self.d2 = value / (1. + self.a[0] + self.a[1])

    def apply(self, value):
        value = np.asarray(value, float)
        if self.b is None:
            return value
        if self.d1 is None:
            self.reset(value)
        d0 = value - self.d1 * self.a[0] - self.d2 * self.a[1]
        out = d0 * self.b[0] + self.d1 * self.b[1] + self.d2 * self.b[2]
        self.d2, self.d1 = self.d1, d0
        return out


class Px4RateController:
    """Body-rate setpoint (FLU rad/s) and gyro -> normalized torque in [-1, 1]."""

    def __init__(self, params):
        k = np.asarray(params["rate_k"], float)
        self.p = k * params["rate_p"]
        self.i = k * params["rate_i"]
        self.d = k * params["rate_d"]
        self.ff = np.asarray(params["rate_ff"], float)
        self.i_limit = np.asarray(params["integrator_limit"], float)
        # MC_*RATE_MAX limits the attitude-controller path; direct rate setpoints are not clamped.
        self.rate_max = (np.deg2rad(params["rate_max_deg_s"]) if params.get("clamp_rate_setpoint", False)
                         else np.full(3, np.inf))
        self.dt = 1. / float(params["rate_loop_hz"])
        hz = float(params["rate_loop_hz"])
        self.gyro_filter = LowPass2p(hz, params["gyro_cutoff_hz"])
        self.dgyro_filter = LowPass2p(hz, params["dgyro_cutoff_hz"])
        self.yaw_alpha = 1. - np.exp(-2. * np.pi * params["yaw_torque_cutoff_hz"] * self.dt)
        self.integral = np.zeros(3)
        self.previous_rate = None
        self.yaw_torque = 0.
        self.saturated = np.zeros((2, 3), bool)  # [positive, negative] per axis from the allocator

    def update(self, rate_sp, gyro):
        rate_sp = np.clip(np.asarray(rate_sp, float), -self.rate_max, self.rate_max)
        rate = self.gyro_filter.apply(gyro)
        previous = rate if self.previous_rate is None else self.previous_rate
        rate_d = self.dgyro_filter.apply((rate - previous) / self.dt)
        self.previous_rate = rate
        error = rate_sp - rate
        torque = self.p * error + self.integral - self.d * rate_d + self.ff * rate_sp
        # Integrator: no growth into allocator saturation, attenuated for large errors (400 deg/s).
        i_error = error.copy()
        i_error[self.saturated[0] & (i_error > 0)] = 0.
        i_error[self.saturated[1] & (i_error < 0)] = 0.
        factor = np.clip(1. - (error / np.deg2rad(400.)) ** 2, 0., 1.)
        self.integral = np.clip(self.integral + factor * self.i * i_error * self.dt, -self.i_limit, self.i_limit)
        self.yaw_torque += self.yaw_alpha * (torque[2] - self.yaw_torque)
        torque[2] = self.yaw_torque
        return np.clip(torque, -1., 1.)


class Px4Allocator:
    """Normalized collective/torque -> normalized motor outputs for the plant's mixer."""

    def __init__(self, mixer):
        mix = np.linalg.pinv(np.asarray(mixer, float))  # motors x [T, tx, ty, tz]
        scale_rp = max(np.sqrt(np.sum(mix[:, 1] ** 2) / (np.count_nonzero(mix[:, 1]) / 2)),
                       np.sqrt(np.sum(mix[:, 2] ** 2) / (np.count_nonzero(mix[:, 2]) / 2)))
        self.roll_pitch = mix[:, 1:3] / scale_rp
        self.yaw = mix[:, 3] / np.max(np.abs(mix[:, 3]))
        self.thrust = mix[:, 0] / np.max(mix[:, 0])

    def allocate(self, collective, torque):
        """Return motor outputs in [0, 1] and per-axis saturation flags [positive, negative]."""
        torque = np.asarray(torque, float)
        rp = self.roll_pitch @ torque[:2]
        base = collective * self.thrust
        # Airmode off: lower the collective to fit the top, never raise it; shrink roll/pitch at the bottom.
        base = base - max(0., np.max(base + rp) - 1.)
        need = rp < -1e-12
        rp_scale = float(np.clip(np.min(np.maximum(base[need], 0.) / -rp[need]), 0., 1.)) if need.any() else 1.
        out = base + rp_scale * rp
        yaw = self.yaw * torque[2]
        room = np.where(yaw > 1e-12, (1. - out) / np.where(yaw > 1e-12, yaw, 1.),
                        np.where(yaw < -1e-12, out / np.where(yaw < -1e-12, -yaw, 1.), np.inf))
        yaw_scale = float(np.clip(np.min(room), 0., 1.))
        out = np.clip(out + yaw_scale * yaw, 0., 1.)
        scales = np.array([rp_scale, rp_scale, yaw_scale])
        flags = np.vstack([(scales < 1.) & (torque > 0), (scales < 1.) & (torque < 0)])
        return out, flags
