"""Offline teacher projection and a rate-loop proxy, never a hardware sender.

The nominal force fraction is NOT an ArduPilot throttle calibration. MAVLink
serialization requires an explicit force->normalized-throttle mapping.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

FLIP = np.diag([1., -1., -1.])  # NWU -> NED, and FLU -> FRD


@dataclass(frozen=True)
class RateThrust:
    body_rates_flu_rad_s: np.ndarray
    collective_force_N: float


def project(command, omega, inertia, mixer, rate_gain):
    wrench = mixer@command
    rates = omega + (wrench[1:]-np.cross(omega, inertia*omega))/(inertia*rate_gain)
    return RateThrust(rates, float(wrench[0]))


def apply_proxy(action, omega, inertia, inverse_mixer, rate_gain, maximum, gain_scale=1., return_saturation=False):
    torque = inertia*rate_gain*gain_scale*(action.body_rates_flu_rad_s-omega)+np.cross(omega,inertia*omega)
    raw=inverse_mixer@np.r_[action.collective_force_N,torque]
    command=np.clip(raw,0,maximum)
    saturated=bool(np.any(raw<0) or np.any(raw>maximum))
    return (command,saturated) if return_saturation else command


def attitude_ned_frd(rotation_nwu_flu):
    q = Rotation.from_matrix(FLIP@rotation_nwu_flu@FLIP).as_quat()
    return q[[3,0,1,2]]


def mavlink_packet(action, timestamp_s, force_to_throttle, target_system=1, target_component=1):
    """Build bytes-compatible SET_ATTITUDE_TARGET fields; sends nothing."""
    throttle = float(force_to_throttle(action.collective_force_N))
    if not np.isfinite(throttle) or not 0 <= throttle <= 1:
        raise ValueError('Calibrated normalized throttle must be in [0,1]')
    return dict(time_boot_ms=int(round(timestamp_s*1000)) % 2**32,
                target_system=target_system,target_component=target_component,type_mask=128,
                q=[1.,0.,0.,0.],body_roll_rate=float(action.body_rates_flu_rad_s[0]),
                body_pitch_rate=float(-action.body_rates_flu_rad_s[1]),body_yaw_rate=float(-action.body_rates_flu_rad_s[2]),
                thrust=throttle)


class ProjectedSkill:
    """Sample/hold + delay + different inner gain on the existing MuJoCo plant.

    This models action-layer sensitivity, NOT ArduPilot firmware. Teacher sees
    closed-loop state, projects each motor action, then only the proxy commands
    reach the plant. ArduPilot PID/shaping/mixing/throttle remain different.
    """
    def __init__(self, skill, hz=200., delay_s=0., gain_scale=1., rate_limit=20.):
        self.skill, self.period, self.delay = skill, 1/hz, delay_s
        self.gain_scale, self.limit = gain_scale, rate_limit
        self.next_update = -np.inf
        self.queue=[]
        self.action=RateThrust(np.zeros(3),skill.mass*9.81)

    def __getattr__(self,name):
        return getattr(self.skill,name)

    def command(self,state,contact):
        motor,sat=self.skill.command(state,contact)
        c=self.cfg['controller']
        gain=2*c['attitude_damping']*c['attitude_frequency_rad_s']
        if state.timestamp >= self.next_update-1e-9:
            a=project(motor,state.omega_body,self.inertia,self.mixer,gain)
            a=RateThrust(np.clip(a.body_rates_flu_rad_s,-self.limit,self.limit),a.collective_force_N)
            self.queue.append((state.timestamp+self.delay,a))
            self.next_update=state.timestamp+self.period
        while self.queue and self.queue[0][0] <= state.timestamp+1e-9:
            _,self.action=self.queue.pop(0)
        u,proxy_saturation=apply_proxy(self.action,state.omega_body,self.inertia,self.inverse_mixer,gain,self.max_motor,self.gain_scale,True)
        # Motor observer uses the command actually applied by the proxy.
        self.skill.previous_motor_command=u.copy()
        return u,bool(sat or proxy_saturation)
