"""Regression coverage for concavities and transforms in the field converter."""
from pathlib import Path
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from rm27_sim.simulation.field_collision import build, convex_parts


@pytest.mark.parametrize('polygon', [
    Polygon([(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)]),
    Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(1,3),(3,3),(3,1)]]),
])
def test_convex_partition_preserves_concavities_and_holes(polygon):
    parts = convex_parts(polygon)
    union = shapely.union_all(parts)
    assert union.symmetric_difference(polygon).area < 1e-9
    assert all(abs(p.area-p.convex_hull.area) < 1e-9 for p in parts)
    assert sum(p.area for p in parts) == pytest.approx(polygon.area)


def test_transformed_ramp_with_hole_has_contact_only_on_its_surface(tmp_path: Path):
    polygon = Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(1,3),(3,3),(3,1)]])
    triangles = np.array([np.asarray(p.exterior.coords)[:3]
                          for p in shapely.constrained_delaunay_triangles(polygon).geoms])
    xy = triangles.reshape(-1,2)
    vertices = np.column_stack([xy, .3*xy[:,0]+.2]).astype('<f4')
    faces = np.arange(len(vertices),dtype='<i4').reshape(-1,3)
    (tmp_path/'assets').mkdir()
    with (tmp_path/'assets/visual_000.msh').open('wb') as f:
        f.write(struct.pack('<4i',len(vertices),0,0,len(faces)))
        vertices.tofile(f);faces.tofile(f)
    root = ET.fromstring('''<mujoco><compiler angle="radian"/><default>
      <default class="collision"><geom contype="1" conaffinity="1"/></default>
      </default><asset><mesh name="visual_000" file="assets/visual_000.msh" inertia="shell"/></asset>
      <worldbody><body pos="2 -3 1" euler="0 0 .4">
      <geom name="visual_000" type="mesh" mesh="visual_000" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    entry = tmp_path/'model.xml';ET.ElementTree(root).write(entry)
    result = build(entry,tmp_path/'generated')
    assert result['collision_geoms'] < 12
    ET.SubElement(root,'include',file='generated/collision_assets.xml')
    world=root.find('worldbody');ET.SubElement(world,'include',file='generated/collision_geoms.xml')
    probe=ET.SubElement(world,'body',name='probe',pos='0 0 10')
    ET.SubElement(probe,'freejoint');ET.SubElement(probe,'geom',type='sphere',size='.01',mass='.1')
    ET.ElementTree(root).write(entry)
    model=mujoco.MjModel.from_xml_path(str(entry));data=mujoco.MjData(model)
    rotation=np.array([[np.cos(.4),-np.sin(.4),0],[np.sin(.4),np.cos(.4),0],[0,0,1]])
    for local, expected_contact in [([2,2,.8],False),([.5,2,.35],True),([.5,2,.45],False)]:
        data.qpos[:3]=rotation@np.array(local)+[2,-3,1]
        mujoco.mj_forward(model,data)
        assert bool(data.ncon) is expected_contact


def test_cad_holes_touching_outer_vertices_remain_open():
    polygon = shapely.from_wkt((Path(__file__).parent/'fixtures/rmuc2027_touching_rings.wkt').read_text())
    assert polygon.is_valid and len(polygon.interiors) == 21
    parts = convex_parts(polygon)
    assert shapely.union_all(parts).symmetric_difference(polygon).area < 1e-10


def test_shared_shells_preserve_pose_and_do_not_box_fill_triangles():
    from rm27_sim.simulation.field_collision import _add_shell
    assets=ET.Element('asset');body=ET.Element('body');cache={}
    triangle=np.array([[0,0,0],[.4,0,0],[0,.3,0]])
    _add_shell(assets,body,cache,triangle,np.array([0.,0.,1.]),'first',.005)
    _add_shell(assets,body,cache,triangle*np.array([-1,1,1])+[2,1,1],
               np.array([0.,0.,1.]),'second',.005)
    assert len(assets)==1 and all(g.get('type')=='mesh' for g in body)
    root=ET.fromstring('<mujoco><default><default class="collision"><geom/></default></default></mujoco>')
    root.append(assets);world=ET.SubElement(root,'worldbody');world.append(body)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    mask=np.ones(6,dtype=np.uint8);gid=np.zeros(1,np.int32)
    for point,expected in [([.05,.05,2],True),([.35,.25,2],False),
                           ([1.95,1.05,2],True),([1.65,1.25,2],False)]:
        distance=mujoco.mj_ray(model,data,np.array(point),np.array([0.,0.,-1.]),mask,1,-1,gid)
        assert (distance>=0)==expected


def test_five_mm_shell_stops_one_metre_drop_and_five_mps_impact():
    from rm27_sim.simulation.field_collision import _add_shell
    root=ET.fromstring('<mujoco><option timestep=".002"/><default><default class="collision"><geom solimp=".95 .99 .001"/></default></default></mujoco>')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    body=ET.SubElement(world,'body');cache={}
    floor=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0.]])
    _add_shell(assets,body,cache,floor,np.array([0.,0.,1.]),'floor',.005)
    probe=ET.SubElement(world,'body',pos='0 0 1');ET.SubElement(probe,'freejoint')
    ET.SubElement(probe,'geom',type='sphere',size='.015',mass='.1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    for _ in range(1000):mujoco.mj_step(model,data)
    assert data.qpos[2]==pytest.approx(.0175,abs=.001) and data.ncon>0
    model.opt.gravity[:]=0;data.qpos[:3]=[0,0,.1];data.qvel[:]=0;data.qvel[2]=-5
    hits=0
    for _ in range(100):
        mujoco.mj_step(model,data);hits+=data.ncon
    assert hits>0 and data.qpos[2]>0


def test_detail_policy_preserves_two_small_lenses_without_filling_the_gap(tmp_path):
    from scipy.spatial import ConvexHull
    # Two 20 mm octahedral lenses: each face is below the default area cutoff.
    one = np.vstack([np.eye(3), -np.eye(3)])*.01
    vertices = np.vstack([one, one+[.1,0,0]]).astype('<f4')
    faces = np.vstack([ConvexHull(one).simplices, ConvexHull(one).simplices+6]).astype('<i4')
    (tmp_path/'assets').mkdir()
    source = tmp_path/'assets/visual_000.msh'
    with source.open('wb') as f:
        f.write(struct.pack('<4i',len(vertices),0,0,len(faces)))
        vertices.tofile(f); faces.tofile(f)
    root = ET.fromstring('''<mujoco><default><default class="collision">
      <geom group="3"/></default></default><asset>
      <mesh name="visual_000" file="assets/visual_000.msh" inertia="shell"/></asset>
      <worldbody><geom name="visual_000" type="mesh" mesh="visual_000"
      group="2" contype="0" conaffinity="0"/></worldbody></mujoco>''')
    entry = tmp_path/'model.xml'; ET.ElementTree(root).write(entry)
    assert build(entry, tmp_path/'before')['collision_geoms'] == 0
    policy = {'sources': {'visual_000': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'mode': 'convex', 'maximum_surface_error_m': .0002}}}
    policy_path = tmp_path/'collision_details.json'
    policy_path.write_text(json.dumps(policy))
    result = build(entry, tmp_path/'after')
    assert result['collision_geoms'] == 2
    ET.SubElement(root, 'include', file='after/collision_assets.xml')
    ET.SubElement(root.find('worldbody'), 'include', file='after/collision_geoms.xml')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'),
                                          assets={'assets/visual_000.msh': source.read_bytes(),
                                                  'after/collision_assets.xml': (tmp_path/'after/collision_assets.xml').read_bytes(),
                                                  'after/collision_geoms.xml': (tmp_path/'after/collision_geoms.xml').read_bytes()})
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    gid = np.zeros(1, np.int32)
    # Sample face interiors: a ray exactly through a triangulation vertex can
    # be rejected by the renderer's triangle-edge roundoff tolerance.
    for x, expected in [(.001, .093), (.101, .093), (.05, -1.)]:
        distance = mujoco.mj_ray(model, data, np.array([x,.002,.1]), np.array([0.,0.,-1.]),
                                 np.array([0,0,0,1,0,0],np.uint8), 1, -1, gid)
        assert distance == pytest.approx(expected, abs=1e-6)
    policy['sources']['visual_000']['sha256'] = 'stale'
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='source changed'):
        build(entry, tmp_path/'stale')
    with pytest.raises(FileNotFoundError):
        build(entry, tmp_path/'missing', detail_config=tmp_path/'missing.json')


def test_convex_detail_rejects_recessed_faces():
    from scipy.spatial import ConvexHull
    from rm27_sim.simulation.field_collision import _add_convex_detail
    points = np.array([[x,y,z] for x in [-.03,.03] for y in [-.03,.03] for z in [-.03,.03]])
    hull = ConvexHull(points)
    tri = hull.simplices[0]
    inset = points[tri].mean(axis=0)-hull.equations[0,:3]*.01
    vertices = np.vstack([points, inset])
    faces = np.vstack([hull.simplices[1:], [tri[0],tri[1],8], [tri[1],tri[2],8], [tri[2],tri[0],8]])
    with pytest.raises(ValueError, match='recess'):
        _add_convex_detail(ET.Element('asset'), ET.Element('body'), vertices, faces, 'recess', .0002)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_backed_skin_aligns_outer_contact_surface_without_moving_unselected_surfaces(sign):
    from rm27_sim.simulation.field_collision import _add_shell, _back_surface
    points = np.array([[-.04,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0.]])
    normal = np.array([0.,0.,sign])
    rule = {'backed_planes': [{'point_m': [0,0,0], 'outward': [0,0,1],
                              'radius_m': .1, 'plane_range_m': [-.0001,.0001]}]}
    shift = _back_surface(points, normal, rule, .005)
    assert shift == pytest.approx([0,0,-.0025])
    assert _back_surface(points+[0,0,.02], normal, rule, .005) == pytest.approx([0,0,0])
    root = ET.fromstring('<mujoco><default><default class="collision"><geom/></default></default></mujoco>')
    assets = ET.SubElement(root, 'asset'); body = ET.SubElement(ET.SubElement(root,'worldbody'), 'body')
    _add_shell(assets, body, {}, points+shift, normal, 'skin', .005)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    distance = mujoco.mj_ray(model,data,np.array([0.,0.,.1]),np.array([0.,0.,-1.]),
                             np.ones(6,np.uint8),1,-1,np.zeros(1,np.int32))
    assert distance == pytest.approx(.1,abs=1e-9)
