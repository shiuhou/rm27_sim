"""Direct actuator access; no invocation of the custom controller."""

from pathlib import Path
import mujoco
import numpy as np
from .actuators import PLANT_ROTORS, pwm_to_thrust
from .frames import attitude_to_ap, specific_force, to_ap

MODEL = Path(__file__).resolve().parents[1] / "models/quadrotor_m1.xml"


class Plant:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )
        if names != PLANT_ROTORS or self.model.opt.timestep != 0.002:
            raise ValueError("plant actuator layout or timestep changed")
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, pwm):
        self.data.ctrl[:] = pwm_to_thrust(pwm)
        mujoco.mj_step(self.model, self.data)
        # Synchronize derived dynamics with the post-integration pose/time.
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        d = self.data
        rotation = d.xmat[self.body].reshape(3, 3)
        state = {
            "timestamp": float(d.time),
            "imu": {
                "gyro": to_ap(d.qvel[3:6]).tolist(),
                "accel_body": specific_force(
                    rotation, d.qacc[:3], self.model.opt.gravity
                ).tolist(),
            },
            "position": to_ap(d.qpos[:3]).tolist(),
            "velocity": to_ap(d.qvel[:3]).tolist(),
            "quaternion": attitude_to_ap(d.qpos[3:7]).tolist(),
        }
        return state

    def truth(self):
        return {
            "position_nwu": self.data.qpos[:3].tolist(),
            "velocity_nwu": self.data.qvel[:3].tolist(),
            "quaternion_flu_to_nwu": self.data.qpos[3:7].tolist(),
            "body_rates_flu": self.data.qvel[3:6].tolist(),
            "command_thrust_n": self.data.ctrl.tolist(),
            "actual_thrust_n": self.data.actuator_force.tolist(),
        }
