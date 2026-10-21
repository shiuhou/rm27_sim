"""Share the large immutable field mesh model, retain exact per-trial physics."""
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

FIELDS = ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'actuator_gear', 'actuator_dynprm',
          'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'site_pos', 'site_quat')


def save_overlay(model, output, template, sha256):
    output, template = Path(output), Path(template)
    np.savez_compressed(output/'model_overlay.npz', **{k:getattr(model, k) for k in FIELDS}, timestep=model.opt.timestep)
    (output/'model_template.json').write_text(json.dumps(dict(path=os.path.relpath(template, output), sha256=sha256), indent=2))


def load_model(source):
    source = Path(source)
    if (source/'model.mjb').exists():
        return mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
    ref = json.loads((source/'model_template.json').read_text())
    path = (source/ref['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Recorded model template hash mismatch')
    model = mujoco.MjModel.from_binary_path(str(path))
    overlay = np.load(source/'model_overlay.npz')
    for key in FIELDS:
        getattr(model, key)[:] = overlay[key]
    model.opt.timestep = float(overlay['timestep'])
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model
