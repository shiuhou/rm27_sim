"""M3 localization estimator unit tests with synthetic sensor sequences.

No MuJoCo is required here: the estimator consumes only IMUSample /
FlowSample / ToFSample. The estimator works in its own LOCAL navigation
frame: x/y start at zero, local +X is the vehicle's initial forward
direction, local yaw starts at 0 even if MuJoCo world yaw is nonzero.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from fly import rotation_to_euler
from m3_estimator import EstimatorConfig, LocalizationEstimator
from m3_sensors import FlowSample, IMUSample, ToFSample

GRAVITY = 9.81
IMU_DT = 0.004  # 250 Hz
AUX_DT = 0.02  # 50 Hz


def imu_sample(
    time: float,
    gyro=(0.0, 0.0, 0.0),
    specific_force=(0.0, 0.0, GRAVITY),
    valid: bool = True,
) -> IMUSample:
    return IMUSample(
        time=time,
        gyro_body=np.asarray(gyro, dtype=float),
        specific_force_body=np.asarray(specific_force, dtype=float),
        valid=valid,
    )


def flow_sample(
    time: float, velocity_xy=(0.0, 0.0), valid: bool = True
) -> FlowSample:
    return FlowSample(
        time=time,
        release_time=time,
        velocity_body_xy=np.asarray(velocity_xy, dtype=float),
        valid=valid,
    )


def tof_sample(time: float, height: float = 1.0, valid: bool = True) -> ToFSample:
    return ToFSample(time=time, release_time=time, height_m=height, valid=valid)


def run_sequence(
    estimator: LocalizationEstimator,
    duration: float,
    gyro=(0.0, 0.0, 0.0),
    specific_force=(0.0, 0.0, GRAVITY),
    flow_velocity=(0.0, 0.0),
    tof_height: float = 1.0,
    flow_valid_window=None,
    tof_valid_window=None,
) -> None:
    """Drive the estimator with constant synthetic sensors."""
    steps = int(round(duration / IMU_DT))
    for k in range(steps + 1):
        t = k * IMU_DT
        estimator.predict(imu_sample(t, gyro, specific_force))
        if k % 5 == 0:  # 50 Hz aux sensors
            flow_ok = (
                True
                if flow_valid_window is None
                else flow_valid_window[0] <= t < flow_valid_window[1]
            )
            tof_ok = (
                True
                if tof_valid_window is None
                else tof_valid_window[0] <= t < tof_valid_window[1]
            )
            estimator.correct_flow(flow_sample(t, flow_velocity, valid=flow_ok))
            estimator.correct_tof(tof_sample(t, tof_height, valid=tof_ok))


class EstimatorBasicsTest(unittest.TestCase):
    def test_stationary_level_stays_at_fixed_point(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 2.0, tof_height=1.0)
        state = est.state(2.0)
        np.testing.assert_allclose(state.position[:2], (0.0, 0.0), atol=1e-9)
        self.assertAlmostEqual(state.position[2], 1.0, places=6)
        np.testing.assert_allclose(state.velocity, (0.0, 0.0, 0.0), atol=1e-9)
        self.assertTrue(state.healthy)

    def test_constant_forward_flow_integrates_position(self) -> None:
        # Constant 1 m/s for 2 s: specific force of unaccelerated level
        # flight is (0,0,g); flow reports 1 m/s body-forward. Estimated x
        # must come from INTEGRATING the corrected velocity, so it is
        # slightly below 2 m due to the complementary-filter transient.
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 2.0, flow_velocity=(1.0, 0.0))
        state = est.state(2.0)
        self.assertGreater(state.position[0], 1.85)
        self.assertLess(state.position[0], 2.0)
        self.assertAlmostEqual(state.velocity[0], 1.0, places=2)

    def test_zero_flow_after_motion_converges_velocity_to_zero(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 1.0, flow_velocity=(1.0, 0.0))
        run_sequence_offset = 1.0
        # Continue with zero flow for 2 more seconds.
        steps = int(round(2.0 / IMU_DT))
        for k in range(1, steps + 1):
            t = run_sequence_offset + k * IMU_DT
            est.predict(imu_sample(t))
            if k % 5 == 0:
                est.correct_flow(flow_sample(t, (0.0, 0.0)))
                est.correct_tof(tof_sample(t, 1.0))
        state = est.state(run_sequence_offset + 2.0)
        self.assertLess(abs(state.velocity[0]), 0.01)

    def test_flow_scale_error_changes_integrated_distance(self) -> None:
        nominal = LocalizationEstimator(EstimatorConfig())
        scaled = LocalizationEstimator(EstimatorConfig())
        # True motion is 1 m/s in both runs; the scaled run's sensor reads
        # 1.02 m/s. Estimated distance must reflect the scale error.
        run_sequence(nominal, 3.0, flow_velocity=(1.0, 0.0))
        run_sequence(scaled, 3.0, flow_velocity=(1.02, 0.0))
        x_nominal = nominal.state(3.0).position[0]
        x_scaled = scaled.state(3.0).position[0]
        self.assertAlmostEqual(x_scaled / x_nominal, 1.02, delta=0.01)
        self.assertGreater(x_scaled - x_nominal, 0.04)

    def test_flow_bias_accumulates_position_error(self) -> None:
        # Vehicle truly stationary; flow bias 0.01 m/s forward. The
        # estimate must drift forward -- this is the failure mode that
        # motivates not trusting the estimate blindly.
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 3.0, flow_velocity=(0.01, 0.0))
        state = est.state(3.0)
        self.assertGreater(state.position[0], 0.015)
        self.assertLess(state.position[0], 0.05)

    def test_gyro_z_bias_accumulates_yaw_error(self) -> None:
        bias = math.radians(0.1)  # 0.1 deg/s
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 2.0, gyro=(0.0, 0.0, bias))
        state = est.state(2.0)
        yaw = rotation_to_euler(state.rotation_body_to_world)[2]
        self.assertAlmostEqual(yaw, bias * 2.0, delta=math.radians(0.01))

    def test_tof_correction_pulls_z_without_truth(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 2.0, tof_height=2.0)
        state = est.state(2.0)
        self.assertAlmostEqual(state.position[2], 2.0, places=3)

    def test_quaternion_remains_normalized_and_finite(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        rng = np.random.default_rng(0)
        for k in range(500):
            t = k * IMU_DT
            est.predict(
                imu_sample(
                    t,
                    gyro=rng.normal(0.0, 0.5, 3),
                    specific_force=(0.0, 0.0, GRAVITY),
                )
            )
            if k % 5 == 0:
                est.correct_flow(flow_sample(t, rng.normal(0.0, 0.2, 2)))
                est.correct_tof(tof_sample(t, 1.0))
        state = est.state(500 * IMU_DT)
        self.assertAlmostEqual(
            float(np.linalg.norm(state.quaternion_wxyz)), 1.0, places=12
        )
        self.assertTrue(np.all(np.isfinite(state.position)))
        self.assertTrue(np.all(np.isfinite(state.velocity)))


class EstimatorHealthTest(unittest.TestCase):
    def test_out_of_order_imu_does_not_reintegrate_time(self) -> None:
        reference = LocalizationEstimator(EstimatorConfig())
        estimator = LocalizationEstimator(EstimatorConfig())
        for current in (reference, estimator):
            current.predict(imu_sample(0.0))
            current.predict(imu_sample(0.01, specific_force=(1.0, 0.0, GRAVITY)))
        for timestamp in (0.01, 0.005, math.nan):
            estimator.predict(imu_sample(timestamp))
        for current in (reference, estimator):
            current.predict(imu_sample(0.02, specific_force=(1.0, 0.0, GRAVITY)))

        expected = reference.state(0.02)
        actual = estimator.state(0.02)
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.velocity, expected.velocity)

    def test_auxiliary_duplicates_and_regressions_are_ignored(self) -> None:
        estimator = LocalizationEstimator(EstimatorConfig())
        estimator.correct_flow(flow_sample(0.02, (1.0, 0.0)))
        estimator.correct_tof(tof_sample(0.02, 1.0))
        before = estimator.state(0.03)
        for timestamp in (0.02, 0.01, math.nan):
            estimator.correct_flow(flow_sample(timestamp, (2.0, 1.0)))
            estimator.correct_tof(tof_sample(timestamp, 2.0))

        after = estimator.state(0.03)
        np.testing.assert_array_equal(after.position, before.position)
        np.testing.assert_array_equal(after.velocity, before.velocity)
        self.assertEqual(after.flow_age, before.flow_age)
        self.assertEqual(after.tof_age, before.tof_age)

    def test_future_samples_and_nonfinite_query_are_unhealthy(self) -> None:
        estimator = LocalizationEstimator(EstimatorConfig())
        estimator.predict(imu_sample(0.02))
        estimator.correct_flow(flow_sample(0.02))
        estimator.correct_tof(tof_sample(0.02))

        self.assertTrue(estimator.state(0.02).healthy)
        for timestamp in (0.01, math.nan, math.inf, -math.inf):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(estimator.state(timestamp).healthy)

    def test_short_flow_dropout_stays_healthy(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 1.0)
        # 0.3 s dropout (limit is 0.5 s).
        steps = int(round(0.3 / IMU_DT))
        for k in range(1, steps + 1):
            t = 1.0 + k * IMU_DT
            est.predict(imu_sample(t))
            if k % 5 == 0:
                est.correct_flow(flow_sample(t, (0.0, 0.0), valid=False))
                est.correct_tof(tof_sample(t, 1.0))
        self.assertTrue(est.state(1.3).healthy)

    def test_long_flow_dropout_becomes_unhealthy(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 1.0)
        steps = int(round(0.6 / IMU_DT))
        for k in range(1, steps + 1):
            t = 1.0 + k * IMU_DT
            est.predict(imu_sample(t))
            if k % 5 == 0:
                est.correct_flow(flow_sample(t, (0.0, 0.0), valid=False))
                est.correct_tof(tof_sample(t, 1.0))
        state = est.state(1.6)
        self.assertFalse(state.healthy)
        self.assertGreater(state.flow_age, 0.5)

    def test_long_tof_dropout_becomes_unhealthy(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 1.0)
        steps = int(round(0.6 / IMU_DT))
        for k in range(1, steps + 1):
            t = 1.0 + k * IMU_DT
            est.predict(imu_sample(t))
            if k % 5 == 0:
                est.correct_flow(flow_sample(t, (0.0, 0.0)))
                est.correct_tof(tof_sample(t, 1.0, valid=False))
        state = est.state(1.6)
        self.assertFalse(state.healthy)
        self.assertGreater(state.tof_age, 0.5)

    def test_nonfinite_state_is_immediately_unhealthy(self) -> None:
        est = LocalizationEstimator(EstimatorConfig())
        run_sequence(est, 0.5)
        est.predict(
            imu_sample(0.5 + IMU_DT, specific_force=(math.nan, 0.0, GRAVITY))
        )
        self.assertFalse(est.state(0.5 + IMU_DT).healthy)


if __name__ == "__main__":
    unittest.main()
