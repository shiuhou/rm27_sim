"""Rail control and optional checks against the generated full-field scene."""
import json
import copy
import os
from pathlib import Path
import socket
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from rm27_sim.integration.base_rail import read_mesh, travel_range, write_subset
from rm27_sim.paths import ROOT, resolve_asset
from rm27_sim.simulation.base_rail import RailLink, RailMotion, controller, load_scene


@pytest.fixture
def motion():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" integrator="implicitfast"/>
      <worldbody><body><joint name="blue_base_slide" type="slide" axis="0 -1 0"
        range="-.3 .3" damping="1"/><geom type="box" size=".1 .1 .1" mass="1"/>
      </body></worldbody><actuator><position name="blue_base_drive"
        joint="blue_base_slide" kp="1000" kv="80" ctrlrange="-.3 .3"
        forcerange="-100 100"/></actuator></mujoco>''')
    return RailMotion(model, mujoco.MjData(model), seed=42)


def test_travel_uses_whole_carriage_footprint():
    limits, _, _ = travel_range([[0, -.42, 0], [0, .42, 0]],
                               [[0, -.12, 0], [0, .12, 0]], [0, -1, 0])
    np.testing.assert_allclose(limits, [-.30, .30])
    with pytest.raises(ValueError):
        travel_range([[0, -.1, 0], [0, .1, 0]], [[0, -.2, 0], [0, .2, 0]], [0, 1, 0])


def test_mesh_split_preserves_triangle_vertices_and_attributes(tmp_path):
    vertices = np.arange(18, dtype=np.float32).reshape(6, 3)
    normals = vertices / 30
    uv = np.arange(12, dtype=np.float32).reshape(6, 2)
    faces = np.array([[0, 1, 2], [2, 3, 0], [3, 4, 5]])
    mask = np.array([True, False, True])
    for selection in [mask, ~mask]:
        path = tmp_path/'part.msh'
        write_subset(path, (vertices, normals, uv, faces), selection)
        v, n, t, f = read_mesh(path)
        np.testing.assert_array_equal(v[f], vertices[faces[selection]])
        np.testing.assert_array_equal(n[f], normals[faces[selection]])
        np.testing.assert_array_equal(t[f], uv[faces[selection]])


def test_servo_clamps_reaches_both_limits_and_seeded_random(motion):
    rng = np.random.default_rng(42)
    for target in [-99, 99, 0, None, None]:
        if target is None:
            assert motion.random_goal() == rng.uniform(*motion.limits)
        else:
            motion.set_goal(target)
        for _ in range(4000):
            previous = motion.command
            motion.tick()
            assert abs(motion.command-previous) <= motion.speed*motion.model.opt.timestep + 1e-12
            mujoco.mj_step(motion.model, motion.data)
            assert -.3005 < motion.data.qpos[0] < .3005
            if motion.reached():
                break
        assert motion.reached()
    with pytest.raises(ValueError):
        motion.set_goal(float('nan'))


def test_udp_keys_release_random_timeout_and_bad_packets(motion):
    link = RailLink(motion)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = ('127.0.0.1', link.port)
    try:
        def send(seq, axis=0, action=None, now=1.):
            client.sendto(json.dumps(dict(seq=seq, axis=axis, action=action)).encode(), address)
            link.poll(now)
        send(0, 1)
        for _ in range(100):
            motion.tick(); mujoco.mj_step(motion.model, motion.data)
        assert motion.command > 0
        send(1)
        assert motion.axis == 0 and motion.goal == motion.command
        send(2, action='random')
        target = motion.goal
        send(3)
        assert motion.goal == target  # idle heartbeat must not cancel random motion
        send(2, action='center')  # old sequence
        assert motion.goal == target
        for payload in [b'[]', b'{', b'null', b'{"seq": 4, "axis": NaN}']:
            client.sendto(payload, address)
            link.poll(1.1)
        assert motion.goal == target
        link.poll(1.5)
        assert motion.axis == 0 and motion.goal == motion.data.qpos[0]
        send(4, -1, now=2.)
        assert motion.axis == -1
        send(5, action='stop', now=2.1)  # focus loss / Space
        assert motion.goal == motion.data.qpos[0]
        send(6, action='center', now=2.2)
        assert motion.goal == 0
        send(7, action='quit', now=2.3)
        assert link.closed
    finally:
        client.close(); link.close()


SCENE = Path(os.environ.get('RM27_BASE_RAIL_SCENE', ROOT/'artifacts/base_rail/model.xml'))


def test_rail_control_coexists_with_other_joints_actuators_and_reset():
    from rm27_sim.simulation.base_rail_control import RailMotion
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <body name="other" pos="0 0 2"><freejoint/>
          <geom type="sphere" size=".05" mass="1"/></body>
        <body name="other_motor" pos="0 0 3"><joint name="other_joint"/>
          <geom type="sphere" size=".05" mass="1"/></body>
        <body><joint name="rail" type="slide" axis="0 -1 0" range="-.3 .3"/>
          <geom type="box" size=".1 .1 .1" mass="1"/></body>
      </worldbody><actuator>
        <motor name="other_drive" joint="other_joint"/>
        <position name="rail_drive" joint="rail" kp="1000" kv="80"/>
      </actuator></mujoco>''')
    data = mujoco.MjData(model)
    data.ctrl[0] = .123
    motion = RailMotion(model, data, speed=.3, joint='rail', actuator='rail_drive')
    assert motion.qadr > 0 and motion.dadr > 0 and motion.actuator == 1
    motion.set_goal(.2)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    motion.tick()
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == 0 and data.ctrl[0] == .123
    for _ in range(2000):
        motion.tick(); mujoco.mj_step(model, data)
    assert motion.reached() and data.ctrl[0] == .123
    mujoco.mj_resetData(model, data)
    motion.reset(seed=12)
    assert motion.command == motion.goal == data.ctrl[1] == 0
    target = motion.random_goal()
    motion.reset(seed=12)
    assert motion.random_goal() == target
    motion.speed = .1
    assert motion.speed == .1
    for invalid in [0, -1, float('inf'), float('nan')]:
        with pytest.raises(ValueError):
            motion.speed = invalid
    with pytest.raises(ValueError, match='limited slide'):
        RailMotion(model, data, joint='rail', actuator='other_drive')


def test_scene_without_manifest_and_nondefault_filename(tmp_path):
    scene = tmp_path/'composed.xml'
    scene.write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    model, manifest = load_scene(scene)
    assert model.ngeom == 1 and manifest == {}


def test_pygame_controller_keyboard_focus_and_udp(monkeypatch):
    """Exercise the real UI loop and socket, supplying OS key/focus states."""
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    import pygame
    from collections import defaultdict
    frame = -1
    messages = []
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(('127.0.0.1', 0))
    server.settimeout(3)

    def receive():
        while True:
            raw, address = server.recvfrom(1024)
            msg = json.loads(raw)
            messages.append(msg)
            server.sendto(json.dumps(dict(position=0, target=0, limits=[-.3, .3])).encode(), address)
            if msg['action'] == 'quit':
                return

    def events():
        nonlocal frame
        frame += 1
        key = {7: pygame.K_r, 9: pygame.K_SPACE, 16: pygame.K_HOME, 17: pygame.K_ESCAPE}.get(frame)
        return [] if key is None else [pygame.event.Event(pygame.KEYDOWN, key=key)]

    def keys():
        state = defaultdict(bool)
        state[pygame.K_RIGHT] = 1 <= frame <= 5
        state[pygame.K_LEFT] = 10 <= frame <= 13 or frame == 15
        return state

    monkeypatch.setattr(pygame.event, 'get', events)
    monkeypatch.setattr(pygame.key, 'get_pressed', keys)
    monkeypatch.setattr(pygame.key, 'get_focused', lambda: frame != 12)
    monkeypatch.setattr(pygame.display, 'get_active', lambda: True)
    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    try:
        controller(server.getsockname()[1])
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert [m['axis'] for m in messages[1:6]] == [1]*5
        assert messages[6]['axis'] == 0
        assert messages[7]['action'] == 'random'
        assert messages[8]['action'] is None
        assert messages[9]['action'] == 'stop'
        assert messages[10]['axis'] == -1
        assert messages[12]['axis'] == 0 and messages[12]['action'] == 'stop'
        assert messages[13]['axis'] == 0  # held key cannot rearm after focus loss
        assert messages[15]['axis'] == -1  # releasing then pressing rearms
        assert messages[16]['action'] == 'center'
        assert messages[17]['action'] == 'quit'
    finally:
        server.close()


@pytest.fixture(scope='module')
def field():
    if not SCENE.exists():
        pytest.skip('Build the optional base rail scene to validate its full geometry')
    model, manifest = load_scene(SCENE)
    return model, manifest


def test_field_retains_each_source_collider_and_visual_triangle(field):
    model, manifest = field
    cfg = manifest['source_config']
    source = resolve_asset(cfg['field_path']).parent
    old = {g.get('name'): g for g in ET.parse(source/'collision_geoms.xml').iter('geom')}
    all_geoms = list(ET.parse(SCENE).iter('geom')) + list(ET.parse(SCENE.parent/'collision_geoms.xml').iter('geom'))
    new = {g.get('name'): g for g in all_geoms if g.get('name') in old}
    assert len([g for g in all_geoms if g.get('name') in old]) == len(old) == len(new)
    moving = set(manifest['moving_collision_names'])
    pivot = np.array(cfg['pivot_m'])
    for name, original in old.items():
        actual = new[name]
        for attr, value in original.attrib.items():
            if attr == 'pos':
                expected = np.fromstring(value, sep=' ') - (pivot if name in moving else 0)
                np.testing.assert_allclose(np.fromstring(actual.get(attr), sep=' '), expected, atol=1e-10)
            else:
                assert actual.get(attr) == value, (name, attr)
    for name, part in manifest['parts'].items():
        original = read_mesh(source/'assets'/f'{name}.msh')
        paths = [SCENE.parent/'assets'/f'{name}.msh']
        if part['fixed_triangles']:
            paths.append(SCENE.parent/'assets'/f'{name}_carriage.msh')
        # Triangle identity includes order and coordinates; splitting must be lossless.
        def triangles(mesh):
            return sorted(map(bytes, mesh[0][mesh[3]].reshape(-1, 9)))
        actual = []
        for path in paths:
            actual.extend(triangles(read_mesh(path)))
        assert sorted(actual) == triangles(original)


def test_field_only_upper_assembly_translates_and_rail_stays_fixed(field):
    model, manifest = field
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    zero = data.geom_xpos.copy()
    carriage = model.body('blue_rail_carriage').id
    selected = model.geom_bodyid == carriage
    axis = np.array(manifest['axis'])
    for q in manifest['travel_range_m'] + [0.]:
        data.qpos[0] = q
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.geom_xpos[selected], zero[selected] + axis*q, atol=1e-10)
        np.testing.assert_allclose(data.geom_xpos[~selected], zero[~selected], atol=1e-10)
        # No field collision may jam the carriage at an endpoint.
        assert data.ncon == 0
        span = np.array(manifest['carriage_span_m']) + q
        guide = manifest['guide_span_m']
        assert span[0] >= guide[0]-1e-10 and span[1] <= guide[1]+1e-10


def test_probe_contacts_moved_lamp_and_armor_without_ghosts(field):
    """Compile exact local collision definitions, with an external free sphere."""
    full, manifest = field
    data = mujoco.MjData(full)
    mujoco.mj_forward(full, data)
    original = ET.parse(SCENE).getroot()
    root = ET.Element('mujoco')
    for tag in ['compiler', 'option', 'default', 'contact', 'actuator']:
        root.append(copy.deepcopy(original.find(tag)))
    world = ET.SubElement(root, 'worldbody')
    for name in ['blue_rail_carriage', 'blue_rail_support']:
        body = copy.deepcopy(original.find(f"worldbody/body[@name='{name}']"))
        for geom in list(body.findall('geom')):
            if geom.get('class') == 'visual':
                body.remove(geom)
        world.append(body)
    fixed = ET.SubElement(world, 'body', name='fixed_base')
    for geom in ET.parse(SCENE.parent/'collision_geoms.xml').iter('geom'):
        xyz = data.geom_xpos[full.geom(geom.get('name')).id]
        if 11 < xyz[0] < 12 and abs(xyz[1]) < 1 and xyz[2] > .9:
            fixed.append(copy.deepcopy(geom))
    asset = ET.SubElement(root, 'asset')
    required = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    for mesh in ET.parse(SCENE.parent/'collision_assets.xml').iter('mesh'):
        if mesh.get('name') in required:
            mesh = copy.deepcopy(mesh)
            if mesh.get('file'):
                mesh.set('file', str(SCENE.parent/mesh.get('file')))
            asset.append(mesh)
    probe = ET.SubElement(world, 'body', name='probe')
    ET.SubElement(probe, 'freejoint', name='probe_joint')
    ET.SubElement(probe, 'geom', name='probe_geom', type='sphere', size='.01', mass='.01')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mujoco.MjData(model)
    jid = model.joint('blue_base_slide').id
    qadr = model.jnt_qposadr[jid]
    padr = model.jnt_qposadr[model.joint('probe_joint').id]
    pid = model.geom('probe_geom').id
    carriage = model.body('blue_rail_carriage').id

    def contacts(q, position):
        d.qpos[qadr] = q
        d.qpos[padr:padr+3] = position
        d.qpos[padr+3:padr+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, d)
        return {int(c.geom[1] if c.geom[0] == pid else c.geom[0])
                for c in d.contact if pid in c.geom and c.dist < 0}

    # Lamp front and upper plate front: the same surface follows the full travel.
    for point, required_name in [([11.482, 0, 1.2424], 'visual_116_detail_0'),
                                 ([11.531, 0, 1.291], None)]:
        for q in [0, *manifest['travel_range_m']]:
            hit = contacts(q, np.array(point) + np.array(manifest['axis'])*q)
            assert any(model.geom_bodyid[g] == carriage for g in hit), (point, q)
            if required_name:
                assert model.geom(required_name).id in hit
        assert not contacts(manifest['travel_range_m'][1], point), point
    # The lower sloped plate still collides at its original location.
    for q in manifest['travel_range_m']:
        hit = contacts(q, [11.445, 0, 1.079])
        assert hit and all(model.geom_bodyid[g] != carriage for g in hit)
