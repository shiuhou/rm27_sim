"""Full-scene regression checks for the integrated target and retained terrain."""
from pathlib import Path
import hashlib
import os
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
if not os.environ.get('RM27_REPAIRED_ARENA'):
    pytest.skip('Set RM27_REPAIRED_ARENA to run repaired arena checks', allow_module_level=True)
from rmuc2027_mjarena.scene import Simulation, load_scene
from rm27_sim.simulation.base_rail_control import RailMotion
from rm27_sim.simulation.base_rail_browser import apply_target_input

ARENA = Path(os.environ['RM27_REPAIRED_ARENA'])
MANIFEST = json.loads((ARENA/'assembly_manifest.json').read_text())

@pytest.fixture(scope='module')
def model():
    return load_scene(ARENA/'model.xml')


def test_terrain_retained_and_repairs_owned_once():
    for name, digest in MANIFEST['source_arena_hashes'].items():
        if name.startswith('closure_') or name.startswith('assets/') and not name.endswith(('visual_109.msh', 'visual_115.msh')):
            assert hashlib.sha256((ARENA/name).read_bytes()).hexdigest() == digest, name
    root = ET.parse(ARENA/'model.xml')
    repairs = list(ET.parse(ARENA/'surface_repair_geoms.xml').iter('geom'))
    repairs += [g for g in root.iter('geom') if g.get('name','').startswith('rmuc2027_repaired_surface_')]
    names = [g.get('name') for g in repairs]
    removed = set(MANIFEST['superseded_lens_repair_names'])
    assert len(names) == len(set(names)) == 18303-107
    assert set(names) == {f'rmuc2027_repaired_surface_{i:05d}' for i in range(18303)}-removed
    carriage = root.find("worldbody/body[@name='blue_rail_carriage']")
    assert set(MANIFEST['moved_repair_names']) <= {g.get('name') for g in carriage.findall('geom')}
    assert len(MANIFEST['moved_repair_names']) == 73
    assert root.find('compiler').get('fusestatic') == 'true'


def test_whole_motion_preserves_fixed_geoms_and_contacts(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    zero=data.geom_xpos.copy();rot=data.geom_xmat.copy()
    moving=model.geom_bodyid==model.body('blue_rail_carriage').id
    assert np.count_nonzero(model.geom_contype|model.geom_conaffinity)==56793
    for q in [*MANIFEST['travel_range_m'],0]:
        data.qpos[0]=q;mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.geom_xpos[moving],zero[moving]+[0,-q,0],atol=1e-9)
        np.testing.assert_allclose(data.geom_xpos[~moving],zero[~moving],atol=1e-9)
        np.testing.assert_allclose(data.geom_xmat,rot,atol=1e-9)
        assert data.ncon==0


def test_endpoints_random_hold_and_reset(model):
    sim=Simulation(model);rail=RailMotion(model,sim.data,speed=.3,seed=42)
    expected=np.random.default_rng(42)
    for goal in [*rail.limits,0,None,None,None,None,None]:
        if goal is None:
            goal=rail.random_goal();assert goal==expected.uniform(*rail.limits)
        else:rail.set_goal(goal)
        for _ in range(3000):
            rail.tick();sim.step()
            assert rail.limits[0]-.0005<=sim.data.qpos[rail.qadr]<=rail.limits[1]+.0005
            if rail.reached():break
        assert rail.reached()
        for _ in range(500):rail.tick();sim.step()
        assert abs(sim.data.qpos[rail.qadr]-goal)<.001
    sim.reset();rail.reset(seed=42)
    assert rail.command==rail.goal==sim.data.ctrl[rail.actuator]==0


@pytest.mark.parametrize('q',[-.3005,0,.3005])
def test_lamp_and_armor_rays_at_all_carriage_positions(model,q):
    data=mujoco.MjData(model);data.qpos[0]=q;mujoco.mj_forward(model,data)
    # mj_ray skips invisible geoms; make collision geoms opaque only for this query.
    alpha=model.geom_rgba[:,3].copy();model.geom_rgba[:,3]=1
    masks=[np.array(x,np.uint8) for x in ([0,0,1,0,0,0],[0,0,0,1,0,0])]
    targets=[('visual_115',[11.4561,.0003,1.079],[-.4617,0,.887],.05,.045),
             ('visual_116',[11.4944,-q,1.2424],[-.968,0,.25],.024,.024),
             ('visual_123',[-11.4577,-.0003,1.079],[.4617,0,.887],.05,.045),
             ('visual_124',[-11.4960,0,1.2424],[.968,0,.25],.024,.024)]
    try:
        for name,center,n,across,along in targets:
            n=np.array(n);n/=np.linalg.norm(n);u=np.array([0,1,0]);v=np.cross(n,u)
            for a in np.linspace(-across,across,21):
                for b in np.linspace(-along,along,21):
                    if name in ('visual_116','visual_124') and a*a+b*b>across*across:continue
                    origin=np.array(center)+a*u+b*v+.15*n;dist=[]
                    for mask in masks:
                        gid=np.array([-1],np.int32)
                        dist.append(mujoco.mj_ray(model,data,origin,-n,mask,1,-1,gid))
                        assert dist[-1]>=0
                    assert abs(dist[1]-dist[0])<.0005,(name,q,a,b,dist)
    finally:model.geom_rgba[:,3]=alpha


def test_browser_input_commands(model):
    sim=Simulation(model);rail=RailMotion(model,sim.data,seed=42)
    apply_target_input(rail,{'ArrowRight'},set());assert rail.axis==1
    for _ in range(100):rail.tick();sim.step()
    apply_target_input(rail,set(),set());assert rail.axis==0 and rail.goal==sim.data.qpos[0]
    apply_target_input(rail,set(),{'KeyT'});target=rail.goal
    apply_target_input(rail,set(),set());assert rail.goal==target
    apply_target_input(rail,set(),{'Home'});assert rail.goal==0
    apply_target_input(rail,set(),{'Space'});assert rail.goal==sim.data.qpos[0]


def test_free_object_contact_at_moved_lamp_and_plate(tmp_path):
    # Compile once with an actual external sphere, preserving the full terrain.
    from rmuc2027_mjarena import Attachment
    probe=tmp_path/'probe.xml'
    probe.write_text('<mujoco><worldbody><body><freejoint name="free"/>'
                     '<geom name="sphere" type="sphere" size=".01" mass=".01"/>'
                     '</body></worldbody></mujoco>')
    m=load_scene(ARENA/'model.xml',[Attachment(probe,'probe/')]);d=mujoco.MjData(m)
    rail=RailMotion(m,d);pid=m.geom('probe/sphere').id
    padr=m.jnt_qposadr[m.joint('probe/free').id]
    dadr=m.jnt_dofadr[m.joint('probe/free').id]
    def hit(q,point):
        mujoco.mj_resetData(m,d);d.qpos[rail.qadr]=q;d.qpos[padr:padr+3]=point
        mujoco.mj_forward(m,d)
        return {int(c.geom[1] if c.geom[0]==pid else c.geom[0]) for c in d.contact if pid in c.geom and c.dist<0}
    for point in [[11.482,0,1.2424],[11.531,0,1.291]]:
        for q in [0,*rail.limits]:
            collisions=hit(q,np.array(point)+[0,-q,0])
            assert any(m.geom_bodyid[g]==m.body('blue_rail_carriage').id for g in collisions)
        assert not hit(rail.limits[1],point)
    for q in rail.limits:
        assert hit(q,[11.445,0,1.079])
    # A 1 m/s sphere impact on the shifted green lens must produce lens contact.
    for q in rail.limits:
        mujoco.mj_resetData(m,d);d.qpos[rail.qadr]=q;rail.reset()
        d.qpos[padr:padr+3]=[11.38,-q,1.2424];d.qvel[dadr:dadr+3]=[1,0,0]
        m.opt.gravity[:]=0
        contacts=0
        for _ in range(150):
            rail.tick();mujoco.mj_step(m,d)
            contacts+=sum(pid in c.geom and m.geom('visual_116_detail_0').id in c.geom for c in d.contact)
        assert contacts>0
