"""Validate and apply explicit simulation dynamics overrides at scene build time."""

import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = {'mass_kg', 'inertia_kg_m2', 'max_rotor_thrust_N',
          'motor_time_constant_s', 'yaw_torque_per_thrust_m'}


def load_profile(path):
    path = Path(path).resolve()
    profile = json.loads(path.read_text())
    if profile.get('schema_version') != 1 or set(profile.get('values', {})) != FIELDS:
        raise ValueError('Calibration requires schema_version 1 and exactly the supported values')
    if profile.get('status') not in ('engineering_estimate', 'fitted_candidate', 'validated'):
        raise ValueError('Missing explicit calibration status')
    evidence = profile.get('evidence', [])
    if profile['status'] != 'engineering_estimate' and not evidence:
        raise ValueError('Fitted or validated profiles need source evidence')
    for item in evidence:
        source = (path.parent / item['path']).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Calibration evidence hash mismatch')
    values = profile['values']
    for key in FIELDS-{'inertia_kg_m2'}:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be a finite positive number')
    inertia = np.asarray(values['inertia_kg_m2'], float)
    if inertia.shape != (3,) or not np.isfinite(inertia).all() or np.any(inertia <= 0):
        raise ValueError('Expected three positive principal inertias')
    if 2*inertia.max() > inertia.sum()+1e-15:
        raise ValueError('Principal inertias violate triangle inequality')
    if profile.get('frame') != 'body axes; principal inertia; COM at model origin':
        raise ValueError('Unsupported calibration frame or COM; requires explicit geometry revision')
    return profile


def with_profile(config, path):
    profile = load_profile(path)
    result = dict(config)
    result['mass_kg'] = profile['values']['mass_kg']
    result['dynamics_override'] = dict(profile['values'])
    result['calibration'] = {'path': str(Path(path).resolve()),
                             'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                             'status': profile['status'], 'evidence': profile['evidence']}
    return result


def verify_compiled(model, body, motors, profile, draw=None):
    """Reject a profile/scene mismatch instead of silently testing other dynamics.

    ``draw`` is a scenario draw whose mass/thrust/motor-lag scales the scene applied on purpose.
    """
    values = profile['values']
    d = draw or {}
    mass, thrust, tau = (d.get(k, 1.) for k in ('mass_scale', 'thrust_scale', 'motor_tau_scale'))
    checks = [(model.body_mass[body], values['mass_kg']*mass),
              (model.body_inertia[body], np.asarray(values['inertia_kg_m2'])*mass),
              (model.actuator_ctrlrange[motors, 1], values['max_rotor_thrust_N']),
              (model.actuator_dynprm[motors, 0], values['motor_time_constant_s']*tau),
              (model.actuator_gear[motors, 5], np.array([1, -1, 1, -1])*values['yaw_torque_per_thrust_m']*thrust)]
    if not all(np.allclose(actual, expected, rtol=1e-12, atol=1e-15) for actual, expected in checks):
        raise ValueError('Scene dynamics differ from the selected calibration profile')
