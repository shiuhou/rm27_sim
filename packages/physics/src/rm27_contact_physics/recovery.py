"""Post-impact detection and immediate-stabilization recovery (no MuJoCo imports).

Detection follows the McGill/Li 1 g rule on the accelerometer residual: measured
specific force minus the thrust the vehicle believes it produces.  Recovery is the
reduced-attitude law of Faessler et al. (global, handles inversion) with zero yaw-rate
command, followed by a PD position hold; Li et al. mode logic is reduced to
"re-erect first, then hold".  Outputs a CTBR command for the existing rate proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


class ImpactDetector:
    """Latch once ||f_meas - (T_est/m) e_z|| exceeds threshold_g * g on ``persistence`` consecutive samples."""

    def __init__(self, threshold_g=1.0, persistence=1):
        if not np.isfinite(threshold_g) or threshold_g <= 0 or int(persistence) < 1:
            raise ValueError("Detection threshold and persistence must be positive")
        self.threshold = float(threshold_g) * GRAVITY
        self.persistence = int(persistence)
        self.count = 0
        self.detected_at = None

    def update(self, timestamp, specific_force_body, thrust_estimate_N, mass_kg):
        if self.detected_at is None:
            residual = np.asarray(specific_force_body, float) - [0., 0., thrust_estimate_N / mass_kg]
            self.count = self.count + 1 if np.linalg.norm(residual) > self.threshold else 0
            if self.count >= self.persistence:
                self.detected_at = float(timestamp)
        return self.detected_at is not None


@dataclass(frozen=True)
class RecoveryGains:
    attitude_rad_s_per_rad: float = 12.0
    yaw_rate_damping: float = 1.0
    position_kp: float = 6.0
    position_kd: float = 4.0
    erect_tilt_rad: float = np.deg2rad(60.)
    max_tilt_rad: float = np.deg2rad(35.)
    max_body_rate_rad_s: float = 15.0
    min_thrust_fraction: float = 0.1


class RecoveryController:
    """Immediate stabilization: re-erect, then hold position at the given setpoint."""

    def __init__(self, mass_kg, max_collective_N, setpoint, gains=RecoveryGains(), *,
                 retreat_direction=None, retreat_until=None, retreat_accel_m_s2=0.):
        self.mass = float(mass_kg)
        self.max_collective = float(max_collective_N)
        self.setpoint = np.asarray(setpoint, float)
        self.gains = gains
        # Optional active retreat: until ``retreat_until`` push away along ``retreat_direction``
        # (gravity compensated, attitude unconstrained) before the position hold takes over.
        self.retreat_direction = None if retreat_direction is None else np.asarray(retreat_direction, float)
        self.retreat_until = retreat_until
        self.retreat_accel = float(retreat_accel_m_s2)

    def command(self, state):
        """Return [collective_N, p, q, r] from a VehicleEstimate-like state (truth or estimate)."""
        g = self.gains
        b3 = state.rotation[:, 2]
        tilt = float(np.arccos(np.clip(b3[2], -1., 1.)))
        if self.retreat_direction is not None and state.timestamp < self.retreat_until:
            desired_acc = self.retreat_accel * self.retreat_direction + [0., 0., GRAVITY]
        elif tilt > g.erect_tilt_rad:
            desired_acc = np.array([0., 0., GRAVITY])
        else:
            desired_acc = (g.position_kp * (self.setpoint - state.position)
                           - g.position_kd * state.velocity + [0., 0., GRAVITY])
            horizontal = desired_acc[:2]
            limit = desired_acc[2] * np.tan(g.max_tilt_rad)
            if desired_acc[2] <= 0:
                desired_acc = np.array([0., 0., GRAVITY])
            elif np.linalg.norm(horizontal) > limit:
                desired_acc[:2] = horizontal * limit / np.linalg.norm(horizontal)
        d = desired_acc / np.linalg.norm(desired_acc)
        cosine = float(np.clip(b3 @ d, -1., 1.))
        axis_world = np.cross(b3, d)
        sine = np.linalg.norm(axis_world)
        if sine < 1e-9:
            # Aligned or exactly inverted: pick body x as the re-erection axis when inverted.
            axis_body = np.zeros(3) if cosine > 0 else np.array([1., 0., 0.])
        else:
            axis_body = state.rotation.T @ (axis_world / sine)
        angle = float(np.arctan2(sine, cosine))
        rates = g.attitude_rad_s_per_rad * angle * axis_body
        rates[2] = -g.yaw_rate_damping * state.omega_body[2]
        rates = np.clip(rates, -g.max_body_rate_rad_s, g.max_body_rate_rad_s)
        # Faessler: thrust only along the projection on the desired direction.
        thrust = self.mass * np.linalg.norm(desired_acc) * max(cosine, 0.)
        thrust = float(np.clip(thrust, g.min_thrust_fraction * self.max_collective, self.max_collective))
        return np.r_[thrust, rates]
