"""Display-only, rigid body-mounted camera for recorded teacher flights."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rm27_sim.paths import ROOT
from .recording import load_model

DEFAULT_MOUNT = ROOT/'configs/contact_fpv_camera.json'


def load_mount(path=DEFAULT_MOUNT):
    return json.loads(Path(path).read_text())


def configure_camera(model, mount, fovy=None):
    """Configure a camera on a private display model; never on the control plant.

    Recorded MJBs cannot grow new objects. Reuse the named display-only overview
    camera; free-camera overview/follow views remain available independently.
    """
    fovy=float(mount['vertical_fov_degrees'] if fovy is None else fovy)
    if not np.isfinite(fovy) or not 1 < fovy < 179:
        raise ValueError('Vertical FOV must be finite and between 1 and 179 degrees')
    position=np.asarray(mount['position_body_m'],dtype=float)
    forward=np.asarray(mount['forward_body'],dtype=float)
    up=np.asarray(mount['up_reference_body'],dtype=float)
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (position,forward,up)):
        raise ValueError('Camera position and axes must be finite 3-vectors')
    if np.linalg.norm(forward)<1e-12:
        raise ValueError('Camera forward axis cannot be zero')
    forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-12:
        raise ValueError('Camera forward and up cannot be parallel')
    right/=np.linalg.norm(right)
    # Native camera axes: image right, image up, optical backward.
    rotation=np.column_stack((right,np.cross(right,forward),-forward))
    quaternion=np.empty(4)
    mujoco.mju_mat2Quat(quaternion,rotation.ravel())
    near=float(mount['near_clip_m'])
    if not np.isfinite(near) or near<=0:
        raise ValueError('Camera near plane must be finite and positive')
    cid=model.camera(mount['camera_slot']).id
    model.cam_bodyid[cid]=model.body(mount['body']).id
    model.cam_mode[cid]=mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.cam_pos[cid]=position
    model.cam_quat[cid]=quaternion
    model.cam_fovy[cid]=fovy
    model.cam_ipd[cid]=0
    # Override any explicit sensor intrinsics on the display camera slot.
    model.cam_sensorsize[cid]=0
    model.vis.map.znear=near/model.stat.extent
    return cid


def restore_state(model,data,row):
    data.time=row[0]
    data.qpos[:],data.qvel[:],data.act[:],data.ctrl[:]=row[1:8],row[8:14],row[14:18],row[18:22]
    mujoco.mj_forward(model,data)


def render_fpv(source,output,*,fovy=None,camera_config=DEFAULT_MOUNT,speed=1.):
    """Export an unannotated MP4 and exact selected-state camera poses."""
    import cv2
    from PIL import Image
    from .io import write_json
    if not np.isfinite(speed) or speed<=0:
        raise ValueError('Positive finite playback speed required')
    source,output=Path(source).resolve(),Path(output).resolve()
    # Hash the original inputs before loading; keep the recorded physics intact.
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    inputs={str(p):digest(p) for p in (source/'trajectory.npz',source/'summary.json',Path(camera_config))}
    if (source/'model.mjb').exists():
        inputs[str(source/'model.mjb')]=digest(source/'model.mjb')
    else:
        ref=json.loads((source/'model_template.json').read_text())
        for p in (source/'model_template.json',source/'model_overlay.npz',(source/ref['path']).resolve()):
            inputs[str(p)]=digest(p)
    model=load_model(source)
    mount=load_mount(camera_config)
    cid=configure_camera(model,mount,fovy)
    trace=np.load(source/'trajectory.npz')['trace']
    if len(trace)<2 or trace.shape[1]<22 or not np.all(np.diff(trace[:,0])>0):
        raise ValueError('A chronological, nonempty single-vehicle trace is required')
    report=json.loads((source/'summary.json').read_text())
    fps=30; width=1280; height=720
    count=int(np.ceil((trace[-1,0]-trace[0,0])/speed*fps))+1
    if count>100000:
        raise ValueError('Preview exceeds 100000 frames; increase playback speed')
    times=np.minimum(trace[0,0]+np.arange(count)*speed/fps,trace[-1,0])
    ids=np.array([int(np.argmin(abs(trace[:,0]-t))) for t in times])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'camera_mount.json',mount)
    data=mujoco.MjData(model)
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(height,model.vis.global_.offheight)
    options=mujoco.MjvOption();options.geomgroup[3]=0;options.sitegroup[:]=0
    writer=cv2.VideoWriter(str(output/'first_person.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    positions=[];rotations=[]
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            def frame(i):
                restore_state(model,data,trace[i])
                renderer.update_scene(data,camera=cid,scene_option=options)
                return renderer.render()
            for i in ids:
                rgb=frame(i)
                writer.write(rgb[:,:,::-1].copy())
                positions.append(data.cam_xpos[cid].copy())
                rotations.append(data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]))
            tc=report.get('terminal',report).get('first_contact',{}).get('time_s',trace[len(trace)//2,0])
            samples=dict(start=trace[0,0],approach=max(trace[0,0],tc-1),contact=tc,
                         recovery=min(trace[-1,0],tc+1.6),returning=(tc+trace[-1,0])/2,finish=trace[-1,0])
            snapshots={}
            for name,t in samples.items():
                i=int(np.argmin(abs(trace[:,0]-t)))
                Image.fromarray(frame(i)).save(output/f'{name}.png')
                snapshots[name]=dict(trace_index=i,simulation_time_s=float(trace[i,0]))
    finally:
        writer.release()
    np.savez_compressed(output/'camera_poses.npz',video_time_s=np.arange(count)/fps,
        simulation_time_s=trace[ids,0],trace_indices=ids,position_world_m=positions,
        R_world_from_camera=rotations)
    f=height/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
    actual_fovy=float(model.cam_fovy[cid])
    unchanged=all(digest(Path(p))==h for p,h in inputs.items())
    result=dict(passed=unchanged,scope='headless rendering of recorded states; no new flight or SITL',
        source=str(source),source_flight_passed=report.get('passed'),source_sha256=inputs,
        source_code_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('viewing.py'),Path(__file__).with_name('recording.py'))},
        original_inputs_unchanged=unchanged,video='first_person.mp4',frames=count,fps=fps,
        width=width,height=height,playback_speed=speed,vertical_fov_degrees=actual_fovy,
        K=[[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1]],
        frame_sampling='Nearest recorded state; actual simulation timestamps and source indices saved in camera_poses.npz',
        pose_convention='p_world = R_world_from_camera @ p_camera + position_world_m; camera right/down/forward',
        snapshots=snapshots,mount=mount,overlays=False,
        note='Follows recorded body attitude including roll; no target tracking or travel-direction alignment. Return can be flown backwards.')
    write_json(output/'summary.json',result)
    return result
