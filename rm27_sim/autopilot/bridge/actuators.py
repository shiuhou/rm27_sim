"""AP_MotorsMatrix Quad X outputs → existing XML actuator thrust commands."""

import numpy as np

AP_ROTORS = ("rotor_fr", "rotor_rl", "rotor_fl", "rotor_rr")
PLANT_ROTORS = ("rotor_fl", "rotor_fr", "rotor_rr", "rotor_rl")
PWM_MIN, PWM_MAX = 1000, 2000
MAX_THRUST = 8.0


def pwm_to_thrust(pwm):
    values = np.asarray(pwm[:4], dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("four finite PWM outputs required")
    ap_thrust = MAX_THRUST * np.clip((values - PWM_MIN) / (PWM_MAX - PWM_MIN), 0, 1)
    return ap_thrust[[2, 0, 3, 1]]
