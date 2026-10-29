"""Re-simulate one recorded impact-recovery trial and play it back (MuJoCo viewer or mp4).

The trial is rebuilt from its study config, physical seed and pre-impact case, so the
playback is the same deterministic simulation that produced the trials.jsonl row.
"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .impact_recovery import load_study
from rm27_contact_physics.impact import ImpactCase, run_trial
from rm27_contact_physics.plant import RM27Plant


def pick_row(trials, condition=None, index=None, outcome="RECOVERED", attitude=None, speed=None):
    rows = [json.loads(line) for line in Path(trials).read_text().splitlines()]
    if index is not None:
        return rows[index]
    condition = condition or rows[0]["condition"]
    candidates = [r for r in rows if r["condition"] == condition and (outcome is None or r["outcome"] == outcome)
                  and (attitude is None or r["cell"].get("attitude_mode") == attitude)
                  and (speed is None or abs(r["cell"].get("normal_speed_m_s", r["case"]["normal_speed_m_s"]) - speed) < 1e-9)]
    if not candidates:
        raise ValueError("No trial matches the requested condition/outcome/cell")
    # Representative rather than best-looking: the median recovery time (or the first match).
    timed = [r for r in candidates if r["recovery_time_s"] is not None]
    if timed:
        timed.sort(key=lambda r: r["recovery_time_s"])
        return timed[len(timed) // 2]
    return candidates[0]


def simulate(trials, row):
    summary = json.loads((Path(trials).parent / "summary.json").read_text())
    study = load_study(summary["provenance"]["config"])
    plant = RM27Plant(study["base"], study["sources_plant"], physical_seed=row["physical_seed"],
                      physical_scale=study["cfg"]["physical_scale"],
                      physical_values=study["cfg"].get("physical_values"))
    case = ImpactCase(**{k: tuple(v) if isinstance(v, list) else v for k, v in row["case"].items()})
    result = run_trial(plant, case, study["conditions"][row["condition"]], study["success"],
                       rate_params=study["rate_params"], imu_params=study["imu_params"],
                       imu_seed=row["physical_seed"], trace=True)
    if result["outcome"] != row["outcome"]:
        raise RuntimeError(f"Replay diverged: {result['outcome']} != recorded {row['outcome']}")
    return plant, result


def _frames(plant, trace):
    qpos = []
    for t in trace:
        q = Rotation.from_rotvec(t[7:10]).as_quat()[[3, 0, 1, 2]]
        qpos.append(np.r_[t[1:4], q])
    return trace[:, 0], np.asarray(qpos)


def _markers(scene, plant, result, trace):
    contact = np.argmax(trace[:, -1] > 0)
    setpoint = trace[contact, 1:4] + np.asarray(result["setpoint_offset_m"] or [0., 0., 0.])
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                        setpoint, np.eye(3).ravel(), np.array([0.1, 0.9, 0.2, 0.6], np.float32))
    scene.ngeom += 1
    return trace[contact, 1:4]


def _camera(cam, plant, impact):
    n = np.asarray(plant.physical["target"]["normal"], float)
    cam.lookat[:] = impact + 0.2 * n  # between the impact point and the 0.3 m backoff setpoint
    cam.distance = 0.85
    cam.azimuth = float(np.degrees(np.arctan2(n[1], n[0]))) + 60.  # beside the plate, looking at its face
    cam.elevation = -12.


def view(trials, row, slow=4.0, loop=True):
    import mujoco.viewer
    plant, result = simulate(trials, row)
    m, d = plant.model, plant.data
    times, qpos = _frames(plant, result["trace"])
    with mujoco.viewer.launch_passive(m, d) as handle:
        handle.opt.geomgroup[3] = 1  # CAD collision proxies (guards, feet) carry the aircraft shape
        d.qpos[:7] = qpos[0]
        mujoco.mj_forward(m, d)
        impact = qpos[np.argmax(result["trace"][:, -1] > 0), :3]
        _camera(handle.cam, plant, impact)
        while handle.is_running():
            start = time.perf_counter()
            for t, q in zip(times, qpos):
                if not handle.is_running():
                    break
                d.qpos[:7] = q
                mujoco.mj_forward(m, d)
                with handle.lock():
                    handle.user_scn.ngeom = 0
                    _markers(handle.user_scn, plant, result, result["trace"])
                handle.sync()
                delay = start + t * slow - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if not loop:
                break
            time.sleep(1.)
    return result


def render(trials, row, output, slow=4.0, fps=30, size=(720, 1280)):
    import cv2
    plant, result = simulate(trials, row)
    m, d = plant.model, plant.data
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, size[1]), max(m.vis.global_.offheight, size[0])
    times, qpos = _frames(plant, result["trace"])
    renderer = mujoco.Renderer(m, *size)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    camera = mujoco.MjvCamera()
    _camera(camera, plant, qpos[np.argmax(result["trace"][:, -1] > 0), :3])
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]))
    for k in range(int(times[-1] * slow * fps) + 1):
        i = min(np.searchsorted(times, k / (slow * fps)), len(times) - 1)
        d.qpos[:7] = qpos[i]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera, option)
        _markers(renderer.scene, plant, result, result["trace"])
        frame = renderer.render()
        cv2.putText(frame, f"t={times[i]:.3f}s  x{1 / slow:.2f}  {row['condition']}  {result['outcome']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return result


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Impact replay</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:12px 16px}img{width:100%;height:auto;background:#000;border-radius:6px}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;align-items:center}
button,input{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px;padding:6px 10px;font:inherit}
input{width:6em}#info{color:#aaa;white-space:pre-wrap}</style></head><body><main>
<div class="bar"><button onclick="c('az',-20)">&#8634; left</button><button onclick="c('az',20)">right &#8635;</button>
<button onclick="c('el',-10)">up</button><button onclick="c('el',10)">down</button>
<button onclick="c('dist',-0.1)">zoom in</button><button onclick="c('dist',0.1)">zoom out</button>
<button onclick="c('slow',-1)">faster</button><button onclick="c('slow',1)">slower</button>
<label>trial line <input id="idx" type="number" min="0"></label><button onclick="pick()">load</button></div>
<img src="/stream" alt="impact replay stream"><div id="info"></div></main>
<script>
async function c(k,d){await fetch(`/control?${k}=${d}`);info()}
async function pick(){await fetch(`/select?index=${document.getElementById('idx').value}`);info()}
async function info(){const r=await fetch('/info');document.getElementById('info').textContent=await r.text()}
info();setInterval(info,3000)
</script></body></html>"""


def stream(trials, row, host="127.0.0.1", port=8765, slow=4.0, fps=30, size=(720, 1280)):
    """Serve the replay as an MJPEG stream rendered off-screen (MUJOCO_GL=egl) in one render thread."""
    import cv2
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    lock = threading.Condition()
    shared = dict(jpeg=None, seq=0, row=row, pending=None, slow=slow, d_az=0., d_el=0., d_dist=0., info="")

    def render_loop():
        renderer = camera = None
        while True:
            row_now = shared["row"]
            plant, result = simulate(trials, row_now)
            m, d = plant.model, plant.data
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, size[1])
            m.vis.global_.offheight = max(m.vis.global_.offheight, size[0])
            renderer = mujoco.Renderer(m, *size)
            option = mujoco.MjvOption(); option.geomgroup[3] = 1
            camera = mujoco.MjvCamera()
            times, qpos = _frames(plant, result["trace"])
            _camera(camera, plant, qpos[np.argmax(result["trace"][:, -1] > 0), :3])
            base = (camera.azimuth, camera.elevation, camera.distance)
            shared["info"] = json.dumps({k: row_now[k] for k in ("condition", "cell", "repeat", "outcome", "recovery_time_s")})
            while shared["pending"] is None:
                start = time.perf_counter()
                for k in range(int(times[-1] * shared["slow"] * fps) + fps):
                    if shared["pending"] is not None:
                        break
                    i = min(np.searchsorted(times, k / (shared["slow"] * fps)), len(times) - 1)
                    camera.azimuth, camera.elevation = base[0] + shared["d_az"], base[1] + shared["d_el"]
                    camera.distance = max(0.15, base[2] + shared["d_dist"])
                    d.qpos[:7] = qpos[i]
                    mujoco.mj_forward(m, d)
                    renderer.update_scene(d, camera, option)
                    _markers(renderer.scene, plant, result, result["trace"])
                    frame = renderer.render()
                    cv2.putText(frame, f"t={times[i]:.3f}s  x{1 / shared['slow']:.2f}  {row_now['condition']}  {result['outcome']}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    with lock:
                        shared["jpeg"], shared["seq"] = jpeg.tobytes(), shared["seq"] + 1
                        lock.notify_all()
                    delay = start + (k + 1) / fps - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        start -= delay  # never try to catch up after a slow frame
            renderer.close()
            shared["row"], shared["pending"] = shared["pending"], None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _text(self, body, kind="text/plain; charset=utf-8"):
            data = body.encode()
            self.send_response(200); self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                return self._text(PAGE, "text/html; charset=utf-8")
            if url.path == "/info":
                return self._text(shared["info"])
            if url.path == "/control":
                for key, name in (("az", "d_az"), ("el", "d_el"), ("dist", "d_dist")):
                    if key in q:
                        shared[name] += float(q[key])
                if "slow" in q:
                    shared["slow"] = float(np.clip(shared["slow"] + float(q["slow"]), 1., 32.))
                return self._text("ok")
            if url.path == "/select":
                shared["pending"] = pick_row(trials, index=int(q["index"]))
                return self._text("ok")
            if url.path != "/stream":
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            seen = -1
            try:
                while True:
                    with lock:
                        lock.wait_for(lambda: shared["seq"] != seen, timeout=5.)
                        jpeg, seen = shared["jpeg"], shared["seq"]
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

    threading.Thread(target=render_loop, daemon=True).start()
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"streaming on http://{host}:{port}/  (Ctrl-C to stop)", flush=True)
    server.serve_forever()
