"""One SITL-controlled vehicle in a shared four-vehicle physical scene."""

from pathlib import Path
import sys
import mujoco
import numpy as np

from rm27_sim.paths import ROOT, resolve_asset
BRIDGE_ROOT = Path(__file__).resolve().parents[1] / "autopilot/bridge"
from ..autopilot.bridge.frames import attitude_to_ap, specific_force, to_ap


class MicroPlant:
    def __init__(self, scene, vehicle=1, dynamics_profile=None):
        self.scene = Path(scene).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.name = f"drone_{vehicle}"
        self.body = self.identifier(mujoco.mjtObj.mjOBJ_BODY, self.name)
        if dynamics_profile is None and not np.isclose(self.model.body_mass[self.body], 0.18, rtol=0, atol=1e-9):
            raise ValueError("This flight adapter is for the provisional 180 g vehicle")
        joint = self.identifier(mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}_free")
        self.qa = int(self.model.jnt_qposadr[joint])
        self.va = int(self.model.jnt_dofadr[joint])
        self.motors = np.array(
            [
                self.identifier(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.name}_motor_{i}")
                for i in range(4)
            ]
        )
        self.maximum = self.model.actuator_ctrlrange[self.motors, 1].copy()
        if dynamics_profile is not None:
            from ..experiments.calibration import load_profile, verify_compiled
            verify_compiled(self.model, self.body, self.motors, load_profile(dynamics_profile))
        if (
            self.model.opt.timestep != 0.002
            or self.model.nu != 16
            or self.model.nq != 28
        ):
            raise ValueError("Expected the 500 Hz four-vehicle scene")
        mujoco.mj_forward(self.model, self.data)
        slot = self.identifier(mujoco.mjtObj.mjOBJ_SITE, f"slot_{vehicle}")
        self.origin = self.data.site_xpos[slot].copy()
        self.initial_quaternion = self.qpos[3:7].copy()
        self.inactive = []
        for n in range(1, 5):
            if n == vehicle:
                continue
            j = self.identifier(mujoco.mjtObj.mjOBJ_JOINT, f"drone_{n}_free")
            qa, va = int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
            self.inactive.append((qa, va, self.data.qpos[qa : qa + 3].copy()))
        self.reset()

    def identifier(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Missing {name}")
        return result

    @property
    def qpos(self):
        return self.data.qpos[self.qa : self.qa + 7]

    @property
    def qvel(self):
        return self.data.qvel[self.va : self.va + 6]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, pwm):
        values = np.asarray(pwm[:4], dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Expected four finite PWM values")
        self.data.ctrl[:] = 0
        self.data.ctrl[self.motors] = (
            np.clip((values[[2, 0, 3, 1]] - 1000) / 1000, 0, 1) * self.maximum
        )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        rotation = self.data.xmat[self.body].reshape(3, 3)
        return {
            "timestamp": float(self.data.time),
            "imu": {
                "gyro": to_ap(self.qvel[3:]).tolist(),
                "accel_body": specific_force(
                    rotation,
                    self.data.qacc[self.va : self.va + 3],
                    self.model.opt.gravity,
                ).tolist(),
            },
            "position": to_ap(self.qpos[:3] - self.origin).tolist(),
            "velocity": to_ap(self.qvel[:3]).tolist(),
            "quaternion": attitude_to_ap(self.qpos[3:7]).tolist(),
        }

    def contacts(self):
        result = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.body not in self.model.geom_bodyid[c.geom]:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            result.append(
                {
                    "geoms": [
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                        for g in c.geom
                    ],
                    "distance_m": float(c.dist),
                    "force_N": float(np.linalg.norm(force[:3])),
                }
            )
        return result

    def truth(self):
        return {
            "position_nwu": self.qpos[:3].tolist(),
            "velocity_nwu": self.qvel[:3].tolist(),
            "quaternion_flu_to_nwu": self.qpos[3:7].tolist(),
            "body_rates_flu": self.qvel[3:].tolist(),
            "command_thrust_n": self.data.ctrl[self.motors].tolist(),
            "actual_thrust_n": self.data.actuator_force[self.motors].tolist(),
        }
