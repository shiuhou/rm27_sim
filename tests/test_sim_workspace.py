"""Standalone source checkout must not rely on parent code or training imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_detached_checkout_can_compose_and_step_without_parent_or_rl(tmp_path):
    shutil.copytree(ROOT / 'rm27_sim', tmp_path / 'rm27_sim',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'configs', tmp_path / 'configs')
    script = '''
import importlib.abc, sys, json
from pathlib import Path
import xml.etree.ElementTree as ET
class BlockParent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'rm27', 'rm27_contact_rl', 'torch', 'gymnasium'}:
            raise ImportError('Forbidden parent/training dependency: '+name)
sys.meta_path.insert(0, BlockParent())
from rm27_sim.integration import dock, drone
from rm27_sim.simulation.micro_plant import MicroPlant
from rm27_sim.autopilot.micro_flight import run
from rm27_sim.experiments.route_experiment import RouteExperiment
from rm27_sim.replay.fpv import load_mount
from rm27_sim.replay.viewing import view_episode
from rm27_sim.simulation.live_view import FlightView
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
print('standalone PASS')
'''
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH')}
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'standalone PASS' in result.stdout


def test_explicit_assets_are_resolved_without_parent_fallback(tmp_path, monkeypatch):
    from rm27_sim import paths
    manifest = tmp_path / 'assets.json'
    field = tmp_path / 'field'
    field.mkdir()
    monkeypatch.setattr(paths, 'ASSET_CONFIG', manifest)
    paths.configure({'field': field})
    assert paths.resolve_asset('rm27/simulation/field/rmuc2026_battlefield_mjcf/model.xml') == field/'model.xml'
    assert paths.resolve_asset('configs/scene.json') == paths.ROOT/'configs/scene.json'
