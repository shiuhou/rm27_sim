#!/usr/bin/env python3
"""Minimal MuJoCo quadrotor position and attitude controller.

Coordinate conventions:
* World and body frames are right-handed with +Z up.
* Body +X points through the red nose and body +Y points left.
* MuJoCo free-joint quaternions use [w, x, y, z].
* Rotation matrices map body-frame vectors into the world frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "quadrotor.xml"
ROTOR_NAMES = ("rotor_fl", "rotor_fr", "rotor_rr", "rotor_rl")
# 這些邊界同時保護互動輸入與自動測試，單位皆為 SI（m、s、N、rad）。
MIN_TARGET_HEIGHT = 0.15
MAX_TARGET_HEIGHT = 5.0
MAX_TARGET_XY = 10.0
TAKEOFF_TIME = 2.0


class SimulationSafetyError(RuntimeError):
    """Raised when the simulation leaves its basic safe operating envelope."""


@dataclass(frozen=True)
class ControllerParameters:
    """All tunable controller and safety parameters in SI units."""

    # position_* 產生期望加速度；attitude_* 再把姿態誤差轉成機體力矩。
    gravity: float = 9.81
    position_kp: FloatArray = field(
        default_factory=lambda: np.array([2.0, 2.0, 5.0], dtype=float)
    )
    position_kd: FloatArray = field(
        default_factory=lambda: np.array([2.4, 2.4, 3.2], dtype=float)
    )
    # M1 robustness：純 PD 在質量誤差下會有恆定懸停誤差
    # e = Δm·g/kp_z（+10% 質量 → 0.196 m，實測確認）。垂直通道加上
    # 小積分項可把恆定力偏差（質量誤差、馬達效率誤差）收斂到零。
    # 只作用於 z 軸，並以 max_vertical_integral_accel 限幅防止 windup。
    position_ki_z: float = 1.5
    max_vertical_integral_accel: float = 2.0
    attitude_kp: FloatArray = field(
        default_factory=lambda: np.array([0.8, 0.8, 0.22], dtype=float)
    )
    attitude_kd: FloatArray = field(
        default_factory=lambda: np.array([0.18, 0.18, 0.08], dtype=float)
    )
    max_horizontal_acceleration: float = 4.0
    max_vertical_feedback: float = 5.0
    max_tilt_radians: float = math.radians(30.0)
    max_torque: FloatArray = field(
        default_factory=lambda: np.array([0.65, 0.65, 0.15], dtype=float)
    )


@dataclass
class TargetCommand:
    """Mutable high-level target; keyboard callbacks only modify this object."""

    position: FloatArray
    yaw: float = 0.0
    velocity: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def clamp(self) -> None:
        # 在 target 層限幅，而不是直接篡改 MuJoCo 的真實位置。
        self.position[:2] = np.clip(self.position[:2], -MAX_TARGET_XY, MAX_TARGET_XY)
        self.position[2] = np.clip(
            self.position[2], MIN_TARGET_HEIGHT, MAX_TARGET_HEIGHT
        )
        self.yaw = wrap_angle(self.yaw)


@dataclass(frozen=True)
class ControlTarget:
    position: FloatArray
    velocity: FloatArray
    yaw: float


@dataclass(frozen=True)
class QuadrotorState:
    position: FloatArray
    velocity_world: FloatArray
    quaternion_wxyz: FloatArray
    rotation_body_to_world: FloatArray
    angular_velocity_body: FloatArray


@dataclass(frozen=True)
class ControlOutput:
    motor_thrusts: FloatArray
    total_thrust: float
    body_torque: FloatArray
    desired_rotation: FloatArray


@dataclass(frozen=True)
class SimulationResult:
    passed: bool
    target_position: FloatArray
    final_position: FloatArray
    final_velocity: FloatArray
    final_euler_rpy: FloatArray
    motor_thrusts: FloatArray
    position_error: float
    horizontal_error: float
    height_error: float
    velocity_norm: float
    reasons: tuple[str, ...]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalize(vector: FloatArray, *, name: str) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-9:
        raise SimulationSafetyError(f"cannot normalize {name}: norm={norm}")
    return vector / norm


def vee(skew_matrix: FloatArray) -> FloatArray:
    """Map a 3x3 skew-symmetric matrix to its vector representation."""

    return np.array(
        [skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]], dtype=float
    )


def rotation_to_euler(rotation: FloatArray) -> FloatArray:
    """Return ZYX roll, pitch, yaw for reporting only (not for control)."""

    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


class TakeoffProfile:
    """Smoothstep altitude command with zero endpoint velocity."""

    def __init__(self, start_position: FloatArray, duration: float = TAKEOFF_TIME):
        self.start_position = np.asarray(start_position, dtype=float).copy()
        self.duration = duration

    def evaluate(self, sim_time: float, command: TargetCommand) -> ControlTarget:
        target_position = command.position.copy()
        target_velocity = command.velocity.copy()
        if sim_time < self.duration:
            phase = float(np.clip(sim_time / self.duration, 0.0, 1.0))
            # 三次 smoothstep s=3t²-2t³；兩端速度皆為零，避免起飛瞬間跳變。
            smooth = phase * phase * (3.0 - 2.0 * phase)
            # smooth_rate 是 ds/dt，讓速度前饋與同一條高度軌跡一致。
            smooth_rate = 6.0 * phase * (1.0 - phase) / self.duration
            height_delta = command.position[2] - self.start_position[2]
            target_position[2] = self.start_position[2] + height_delta * smooth
            target_velocity[2] = height_delta * smooth_rate
        return ControlTarget(target_position, target_velocity, command.yaw)


class QuadrotorController:
    """Cascaded position/SO(3)-attitude controller with a four-rotor mixer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        parameters: ControllerParameters | None = None,
        *,
        body_name: str = "quadrotor",
        joint_name: str = "root",
        actuator_names: tuple[str, ...] = ROTOR_NAMES,
        control_dt_s: float | None = None,
    ) -> None:
        self.model = model
        self.parameters = parameters or ControllerParameters()
        self.control_dt_s = float(model.opt.timestep if control_dt_s is None else control_dt_s)
        if not math.isfinite(self.control_dt_s) or self.control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        self.root_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if self.body_id < 0 or self.root_joint_id < 0:
            raise ValueError(f"model must contain body {body_name!r} and free joint {joint_name!r}")
        if (model.jnt_type[self.root_joint_id] != mujoco.mjtJoint.mjJNT_FREE
                or model.jnt_bodyid[self.root_joint_id] != self.body_id):
            raise ValueError("controller joint must be the selected body's free joint")
        self.qpos_address = int(model.jnt_qposadr[self.root_joint_id])
        self.mass = float(model.body_mass[self.body_id])
        # 垂直誤差積分器（單位：m·s），只服務 altitude channel。
        self._vertical_error_integral = 0.0

        if len(actuator_names) != 4 or len(set(actuator_names)) != 4:
            raise ValueError('Expected four distinct named rotor actuators')
        self.actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in actuator_names
        ])
        if np.any(self.actuator_ids < 0):
            raise ValueError(f'Missing rotor actuators: {actuator_names}')
        self.minimum_thrusts = model.actuator_ctrlrange[self.actuator_ids, 0].astype(float).copy()
        self.maximum_thrusts = model.actuator_ctrlrange[self.actuator_ids, 1].astype(float).copy()
        if not np.all(model.actuator_ctrllimited[self.actuator_ids]):
            raise ValueError('All rotor actuators must have control limits')
        if not np.allclose(self.minimum_thrusts, 0.0):
            raise ValueError("rotor minimum thrust must be zero")

        self.allocation_matrix = self._build_allocation_matrix()
        if np.linalg.matrix_rank(self.allocation_matrix) != 4:
            raise ValueError("rotor allocation matrix is singular")
        self.inverse_allocation_matrix = np.linalg.inv(self.allocation_matrix)

    def reset_vertical_integrator(self) -> None:
        """Clear the altitude integrator (called on simulation reset)."""

        self._vertical_error_integral = 0.0

    def _build_allocation_matrix(self) -> FloatArray:
        """Map FL/FR/RR/RL thrusts to [force_z, torque_x/y/z]."""

        # 每一欄代表一顆旋翼的貢獻。由 τ=r×F 可得
        # τx=y·Fz、τy=-x·Fz；τz 則讀取 XML gear 的反作用力矩係數。
        allocation = np.zeros((4, 4), dtype=float)
        for column, actuator_id in enumerate(self.actuator_ids):
            rotor_name = self.model.actuator(actuator_id).name
            if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_SITE:
                raise ValueError(f"{rotor_name} must use a rotor site transmission")
            site_id = int(self.model.actuator_trnid[actuator_id, 0])
            if self.model.site_bodyid[site_id] != self.body_id:
                raise ValueError(f"{rotor_name} site must belong to the controlled body")
            if not np.allclose(self.model.site_quat[site_id], [1, 0, 0, 0]):
                raise ValueError(f"{rotor_name} site axes must match the body axes")
            x_position, y_position = self.model.site_pos[site_id, :2]
            gear = self.model.actuator_gear[actuator_id]
            if not np.allclose(gear[:2], 0.0) or not math.isclose(gear[2], 1.0):
                raise ValueError(f"{rotor_name} must apply unit local +Z force")
            allocation[:, column] = (
                1.0,
                y_position,
                -x_position,
                gear[5],
            )
        return allocation

    def read_state(self, data: mujoco.MjData) -> QuadrotorState:
        # freejoint 的 qpos 固定是 [x, y, z, qw, qx, qy, qz]。
        qpos = data.qpos[self.qpos_address : self.qpos_address + 7]
        position = qpos[:3].copy()
        quaternion = qpos[3:7].copy()  # MuJoCo order: w, x, y, z.
        quaternion = normalize(quaternion, name="quaternion")

        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation_flat, quaternion)
        rotation = rotation_flat.reshape(3, 3)

        # mj_objectVelocity 回傳 [角速度, 線速度]；flg_local=0 表示世界座標。
        spatial_velocity_world = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            spatial_velocity_world,
            0,
        )
        angular_velocity_world = spatial_velocity_world[:3]
        linear_velocity_world = spatial_velocity_world[3:]
        # SO(3) 姿態控制使用機體角速度，因此以 Rᵀ 將世界向量轉回 body frame。
        angular_velocity_body = rotation.T @ angular_velocity_world

        state_values = np.concatenate(
            (
                position,
                linear_velocity_world,
                quaternion,
                rotation.ravel(),
                angular_velocity_body,
            )
        )
        if not np.all(np.isfinite(state_values)):
            raise SimulationSafetyError("state contains NaN or Inf")
        return QuadrotorState(
            position,
            linear_velocity_world.copy(),
            quaternion,
            rotation,
            angular_velocity_body,
        )

    def position_control(
        self, state: QuadrotorState, target: ControlTarget
    ) -> tuple[float, FloatArray]:
        """Return collective thrust and desired body-to-world rotation."""

        parameters = self.parameters
        # 外迴路是逐軸 PD：a_fb = Kp(p_d-p) + Kd(v_d-v)。
        position_error = target.position - state.position
        velocity_error = target.velocity - state.velocity_world
        feedback_acceleration = (
            parameters.position_kp * position_error
            + parameters.position_kd * velocity_error
        )
        # 垂直積分項：以控制更新週期積分高度誤差（缺省為 model timestep），
        # 穩態時補償恆定力偏差，transient 時幾乎不影響原本的 PD 行為。
        integral_limit = (
            parameters.max_vertical_integral_accel / parameters.position_ki_z
        )
        self._vertical_error_integral = float(
            np.clip(
                self._vertical_error_integral + position_error[2] * self.control_dt_s,
                -integral_limit,
                integral_limit,
            )
        )
        feedback_acceleration[2] += (
            parameters.position_ki_z * self._vertical_error_integral
        )
        feedback_acceleration[2] = np.clip(
            feedback_acceleration[2],
            -parameters.max_vertical_feedback,
            parameters.max_vertical_feedback,
        )

        horizontal = feedback_acceleration[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm > parameters.max_horizontal_acceleration:
            feedback_acceleration[:2] *= (
                parameters.max_horizontal_acceleration / horizontal_norm
            )

        # 加回重力補償；定點時期望加速度約為 [0, 0, g]。
        desired_acceleration = feedback_acceleration + np.array(
            [0.0, 0.0, parameters.gravity]
        )
        desired_acceleration[2] = max(desired_acceleration[2], 0.5)

        # Enforce the tilt limit by bounding lateral acceleration relative to +Z.
        tilt_lateral_limit = desired_acceleration[2] * math.tan(
            parameters.max_tilt_radians
        )
        lateral_norm = float(np.linalg.norm(desired_acceleration[:2]))
        if lateral_norm > tilt_lateral_limit:
            desired_acceleration[:2] *= tilt_lateral_limit / lateral_norm

        desired_force_world = self.mass * desired_acceleration
        # 期望合力方向就是機體 +Z；yaw heading 再決定水平面的機頭方向。
        desired_body_z = normalize(desired_force_world, name="desired force")
        heading = np.array([math.cos(target.yaw), math.sin(target.yaw), 0.0])
        desired_body_y = normalize(
            np.cross(desired_body_z, heading), name="desired body Y"
        )
        desired_body_x = np.cross(desired_body_y, desired_body_z)
        desired_rotation = np.column_stack(
            (desired_body_x, desired_body_y, desired_body_z)
        )

        # 旋翼只能沿目前 body +Z 推；投影可避免傾斜時錯把世界 +Z 力直接當推力。
        collective_thrust = float(
            desired_force_world @ state.rotation_body_to_world[:, 2]
        )
        collective_thrust = float(
            np.clip(collective_thrust, 0.0, np.sum(self.maximum_thrusts))
        )
        return collective_thrust, desired_rotation

    def attitude_control(
        self, state: QuadrotorState, desired_rotation: FloatArray
    ) -> FloatArray:
        """SO(3) attitude feedback; Euler angles are never used for control."""

        rotation = state.rotation_body_to_world
        # e_R = vee((R_dᵀR - RᵀR_d)/2)，直接在 SO(3) 上比較姿態，
        # 可避開 Euler angle 的萬向節鎖與角度跨 ±π 問題。
        attitude_error_matrix = 0.5 * (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = vee(attitude_error_matrix)
        body_torque = (
            -self.parameters.attitude_kp * attitude_error
            - self.parameters.attitude_kd * state.angular_velocity_body
        )
        return np.clip(
            body_torque, -self.parameters.max_torque, self.parameters.max_torque
        )

    def mix_controls(self, total_thrust: float, body_torque: FloatArray) -> FloatArray:
        """Allocate desired wrench to four non-negative rotor thrusts."""

        # 反解 allocation matrix：期望 [Fz, τx, τy, τz] → 四顆旋翼推力。
        desired_wrench = np.concatenate(([total_thrust], body_torque))
        if desired_wrench.shape != (4,) or not np.all(np.isfinite(desired_wrench)):
            raise SimulationSafetyError("desired wrench is invalid")
        motor_thrusts = self.inverse_allocation_matrix @ desired_wrench
        motor_thrusts = np.clip(
            motor_thrusts, self.minimum_thrusts, self.maximum_thrusts
        )
        if motor_thrusts.shape != (4,) or not np.all(np.isfinite(motor_thrusts)):
            raise SimulationSafetyError("motor thrust command is invalid")
        return motor_thrusts

    def update(
        self,
        data: mujoco.MjData,
        target: ControlTarget,
        measured_state: QuadrotorState | None = None,
    ) -> ControlOutput:
        # M1 measurement noise 由 measured_state 注入；預設仍讀 ground truth。
        state = measured_state if measured_state is not None else self.read_state(data)
        total_thrust, desired_rotation = self.position_control(state, target)
        body_torque = self.attitude_control(state, desired_rotation)
        motor_thrusts = self.mix_controls(total_thrust, body_torque)
        return ControlOutput(
            motor_thrusts, total_thrust, body_torque, desired_rotation
        )


def load_simulation(
    model_path: Path,
) -> tuple[mujoco.MjModel, mujoco.MjData, QuadrotorController]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = QuadrotorController(model)
    return model, data, controller


def make_command(initial_position: FloatArray, target_height: float) -> TargetCommand:
    command = TargetCommand(initial_position.copy())
    command.position[2] = target_height
    command.clamp()
    return command


def check_runtime_safety(
    state: QuadrotorState,
    motor_thrusts: FloatArray,
    controller: QuadrotorController,
    sim_time: float,
) -> None:
    if state.position[2] < -0.25 or state.position[2] > 10.0:
        raise SimulationSafetyError(f"unsafe altitude: {state.position[2]:.3f} m")
    if sim_time > 0.5 and state.rotation_body_to_world[2, 2] < math.cos(
        math.radians(70.0)
    ):
        raise SimulationSafetyError("vehicle has flipped beyond 70 degrees")
    if not np.all(np.isfinite(motor_thrusts)):
        raise SimulationSafetyError("motor commands contain NaN or Inf")
    tolerance = 1e-9
    if np.any(motor_thrusts < controller.minimum_thrusts - tolerance) or np.any(
        motor_thrusts > controller.maximum_thrusts + tolerance
    ):
        raise SimulationSafetyError("motor command outside actuator limits")


def step_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: QuadrotorController,
    target: ControlTarget,
    measured_state: QuadrotorState | None = None,
) -> ControlOutput:
    # 一個 timestep 的固定順序：讀狀態/算控制 → 寫 actuator → 安全檢查 → 積分。
    # measured_state 只影響控制器看到的狀態；安全檢查永遠用 ground truth。
    output = controller.update(data, target, measured_state=measured_state)
    data.ctrl[controller.actuator_ids] = output.motor_thrusts
    check_runtime_safety(
        controller.read_state(data), output.motor_thrusts, controller, data.time
    )
    mujoco.mj_step(model, data)
    return output


def validate_result(
    controller: QuadrotorController,
    data: mujoco.MjData,
    target: ControlTarget,
    motor_thrusts: FloatArray,
    runtime_error: str | None = None,
) -> SimulationResult:
    reasons: list[str] = []
    try:
        state = controller.read_state(data)
    except SimulationSafetyError as error:
        zero = np.zeros(3, dtype=float)
        return SimulationResult(
            False,
            target.position.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            motor_thrusts.copy(),
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            (str(error),),
        )

    position_delta = target.position - state.position
    position_error = float(np.linalg.norm(position_delta))
    horizontal_error = float(np.linalg.norm(position_delta[:2]))
    height_error = abs(float(position_delta[2]))
    velocity_norm = float(np.linalg.norm(state.velocity_world))
    euler = rotation_to_euler(state.rotation_body_to_world)

    values = np.concatenate(
        (state.position, state.velocity_world, euler, motor_thrusts)
    )
    if not np.all(np.isfinite(values)):
        reasons.append("final state or motor thrust contains NaN/Inf")
    if state.position[2] <= 0.5:
        reasons.append(f"final altitude {state.position[2]:.3f} m is not above 0.5 m")
    if height_error >= 0.10:
        reasons.append(f"height error {height_error:.3f} m is not below 0.10 m")
    if horizontal_error >= 0.20:
        reasons.append(
            f"horizontal error {horizontal_error:.3f} m is not below 0.20 m"
        )
    if abs(euler[0]) > math.radians(15.0) or abs(euler[1]) > math.radians(15.0):
        reasons.append("final roll or pitch exceeds 15 degrees")
    if np.any(motor_thrusts < controller.minimum_thrusts) or np.any(
        motor_thrusts > controller.maximum_thrusts
    ):
        reasons.append("final motor thrust is outside actuator limits")
    if runtime_error:
        reasons.append(runtime_error)

    return SimulationResult(
        not reasons,
        target.position.copy(),
        state.position,
        state.velocity_world,
        euler,
        motor_thrusts.copy(),
        position_error,
        horizontal_error,
        height_error,
        velocity_norm,
        tuple(reasons),
    )


def simulate_headless(
    model_path: Path = DEFAULT_MODEL_PATH,
    duration: float = 8.0,
    target_height: float = 1.2,
) -> SimulationResult:
    """Run the complete controller without constructing a viewer."""

    if duration <= 0.0:
        raise ValueError("headless duration must be greater than zero")
    if not MIN_TARGET_HEIGHT <= target_height <= MAX_TARGET_HEIGHT:
        raise ValueError(
            f"target height must be in [{MIN_TARGET_HEIGHT}, {MAX_TARGET_HEIGHT}] m"
        )

    model, data, controller = load_simulation(Path(model_path))
    initial_state = controller.read_state(data)
    command = make_command(initial_state.position, target_height)
    profile = TakeoffProfile(initial_state.position)
    output = ControlOutput(np.zeros(4), 0.0, np.zeros(3), np.eye(3))
    target = profile.evaluate(0.0, command)
    runtime_error: str | None = None

    try:
        while data.time < duration:
            target = profile.evaluate(data.time, command)
            output = step_controller(model, data, controller, target)
    except SimulationSafetyError as error:
        runtime_error = f"runtime safety failure at {data.time:.3f} s: {error}"

    final_target = profile.evaluate(data.time, command)
    return validate_result(
        controller,
        data,
        final_target,
        output.motor_thrusts,
        runtime_error,
    )


def print_target(command: TargetCommand) -> None:
    position = ", ".join(f"{value:.2f}" for value in command.position)
    print(f"target position: [{position}], yaw: {math.degrees(command.yaw):.1f} deg")


def print_result(result: SimulationResult) -> None:
    def vector(values: FloatArray, precision: int = 3) -> str:
        return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"

    print("Simulation result")
    print("-----------------")
    print(f"target_position: {vector(result.target_position)}")
    print(f"final_position:  {vector(result.final_position)}")
    print(f"position_error:  {result.position_error:.3f} m")
    print(f"horizontal_error:{result.horizontal_error: .3f} m")
    print(f"height_error:    {result.height_error:.3f} m")
    print(f"velocity_norm:   {result.velocity_norm:.3f} m/s")
    print(
        "roll_pitch_yaw: "
        + vector(np.degrees(result.final_euler_rpy), precision=2)
        + " deg"
    )
    print(f"motor_thrusts:   {vector(result.motor_thrusts)} N")
    if result.passed:
        print("PASS")
    else:
        for reason in result.reasons:
            print(f"FAIL: {reason}")


def run_viewer(
    model_path: Path,
    duration: float,
    target_height: float,
    realtime_factor: float,
    print_rate: float,
) -> int:
    # Importing viewer lazily keeps headless execution independent of a display.
    import mujoco.viewer

    model, data, controller = load_simulation(model_path)
    initial_state = controller.read_state(data)
    command = make_command(initial_state.position, target_height)
    profile = TakeoffProfile(initial_state.position)
    print_target(command)

    next_sync_time = 0.0
    next_print_time = 0.0
    sync_period = 1.0 / 60.0
    try:
        # Observation/camera only. Flight input belongs to pygame_controller.py.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if data.time >= duration:
                    break

                wall_start = time.perf_counter()
                target = profile.evaluate(data.time, command)
                output = step_controller(model, data, controller, target)

                if data.time >= next_sync_time:
                    viewer.sync()
                    next_sync_time = data.time + sync_period
                if print_rate > 0.0 and data.time >= next_print_time:
                    state = controller.read_state(data)
                    print(
                        f"t={data.time:6.2f} s  z={state.position[2]:5.2f} m  "
                        f"speed={np.linalg.norm(state.velocity_world):5.2f} m/s  "
                        f"motors={np.round(output.motor_thrusts, 2)}"
                    )
                    next_print_time = data.time + 1.0 / print_rate

                desired_wall_step = model.opt.timestep / realtime_factor
                remaining = desired_wall_step - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    except SimulationSafetyError as error:
        print(f"simulation stopped: {error}", file=sys.stderr)
        return 1
    return 0


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def bounded_height(value: str) -> float:
    height = float(value)
    if not MIN_TARGET_HEIGHT <= height <= MAX_TARGET_HEIGHT:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_TARGET_HEIGHT} and {MAX_TARGET_HEIGHT} m"
        )
    return height


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal MuJoCo quadrotor takeoff and hover simulation."
    )
    parser.add_argument("--trajectory", choices=("hover",), default="hover")
    parser.add_argument(
        "--headless", action="store_true", help="run without constructing a viewer"
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=None,
        help="simulation seconds (default: 8 headless, unlimited with GUI)",
    )
    parser.add_argument(
        "--target-height", type=bounded_height, default=1.2, metavar="METERS"
    )
    parser.add_argument(
        "--realtime-factor", type=positive_float, default=1.0, metavar="FACTOR"
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=1.0,
        metavar="HZ",
        help="GUI status rate; use 0 to disable",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH, metavar="XML"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"model file does not exist: {model_path}")
    if args.print_rate < 0.0:
        parser.error("--print-rate cannot be negative")
    if args.headless:
        duration = args.duration if args.duration is not None else 8.0
        result = simulate_headless(model_path, duration, args.target_height)
        print_result(result)
        return 0 if result.passed else 1

    duration = args.duration if args.duration is not None else math.inf
    return run_viewer(
        model_path,
        duration,
        args.target_height,
        args.realtime_factor,
        args.print_rate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
