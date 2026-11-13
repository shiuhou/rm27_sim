"""Shared simulation stays independent of task algorithms and training imports."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_detached_checkout_can_compose_and_step_without_parent_non_rl_or_rl(tmp_path):
    shutil.copytree(ROOT / 'rm27_sim', tmp_path / 'rm27_sim',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'configs', tmp_path / 'configs')
    shutil.copytree(ROOT / 'packages/physics/src/rm27_contact_physics',
                    tmp_path / 'rm27_contact_physics',
                    ignore=shutil.ignore_patterns('__pycache__'))
    script = '''
import importlib.abc, sys, json
from pathlib import Path
import xml.etree.ElementTree as ET
class BlockParent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'rm27', 'rm27_non_rl', 'rm27_contact_rl', 'torch', 'gymnasium'}:
            raise ImportError('Forbidden task/training dependency: '+name)
sys.meta_path.insert(0, BlockParent())
from rm27_sim.integration import dock, drone
from rm27_sim.simulation.micro_plant import MicroPlant
from rm27_sim.autopilot.micro_flight import run
from rm27_sim.experiments.route_experiment import RouteExperiment
from rm27_sim.replay.fpv import load_mount
from rm27_sim.replay.viewing import view_episode
from rm27_sim.simulation.live_view import FlightView
from rm27_sim.simulation.asm2 import Plant, ContactScorer, load_scene
from rm27_sim.experiments.asm2_sensor_recording import SensorRecorder
from rm27_sim.experiments.asm2_recovery_fixture import load_fixture
from rm27_contact_physics.plant import RM27Plant
import mujoco, numpy as np
root=ET.fromstring('<mujoco><option timestep=".002" integrator="RK4"/><asset/><worldbody/></mujoco>')
body=ET.SubElement(root.find('worldbody'),'body',name='dock')
for name,p,size in dock.boxes():
    ET.SubElement(body,'geom',name=name,type='box',pos=' '.join(map(str,p)),size=' '.join(map(str,size)))
cfg=json.loads(Path('configs/drone.json').read_text())
for i,p in enumerate(dock.slot_centers().values()):
    drone.add(root, f'drone_{i+1}', np.array(p)+[0,0,.021], [1,0,0,0], cfg)
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
data=mujoco.MjData(model)
for _ in range(10): mujoco.mj_step(model,data)
assert np.isfinite(data.qpos).all() and data.time>0
assert load_mount()['body']=='uav'
assert not any(name == 'rm27_non_rl' or name.startswith('rm27_non_rl.') for name in sys.modules)
print('standalone PASS')
'''
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH')}
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'standalone PASS' in result.stdout


def test_historical_asm2_imports_share_canonical_non_rl_modules():
    for name in ('asm2_contact_control', 'asm2_guidance', 'asm2_estimation_integrity',
                 'asm2_sensor_estimator', 'asm2_flight_alignment',
                 'asm2_recovery_initialization', 'asm2_estimation_replay',
                 'asm2_integrated'):
        historical = importlib.import_module(f'rm27_sim.experiments.{name}')
        canonical = importlib.import_module(f'rm27_non_rl.{name}')
        assert historical is canonical


def test_m3_compatibility_import_uses_canonical_non_rl_estimator():
    historical = importlib.import_module('m3_estimator')
    canonical = importlib.import_module('rm27_non_rl.m3_estimator')
    assert historical.LocalizationEstimator is canonical.LocalizationEstimator
    assert historical.EstimatorConfig is canonical.EstimatorConfig


def test_explicit_assets_are_resolved_without_parent_fallback(tmp_path, monkeypatch):
    from rm27_sim import paths
    manifest = tmp_path / 'assets.json'
    field = tmp_path / 'field'
    field.mkdir()
    monkeypatch.setattr(paths, 'ASSET_CONFIG', manifest)
    paths.configure({'field': field})
    assert paths.resolve_asset('rm27/simulation/field/rmuc2026_battlefield_mjcf/model.xml') == field/'model.xml'
    assert paths.resolve_asset('configs/scene.json') == paths.ROOT/'configs/scene.json'


def test_view_loads_compiled_model_before_launch(tmp_path, monkeypatch):
    import mujoco
    import mujoco.viewer
    from rm27_sim.__main__ import main
    scene = tmp_path/'scene.mjb'
    original = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    mujoco.mj_saveModel(original, str(scene))
    loaded = []
    monkeypatch.setattr(mujoco.viewer, 'launch', lambda *, model: loaded.append(model))
    monkeypatch.setattr(sys, 'argv', ['rm27_sim', 'view', str(scene)])
    main()
    assert len(loaded) == 1 and loaded[0].ngeom == original.ngeom == 1
