"""v7: Mesh3d 球体 (光照亮面) + 深度着色骨架 + 不干扰拖动"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import json
from pathlib import Path


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_conv(x) for x in o]
    return o


def icosphere(radius, subdiv=1):
    """生成 icosphere 顶点/面 (细分二十面体)"""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(subdiv):
        verts, faces = _subdivide(verts, faces)
    verts = np.array(verts, dtype=np.float64)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return verts, np.array(faces, dtype=np.int64)


def _subdivide(verts, faces):
    verts = [list(v) for v in verts]
    mid_cache = {}
    def mid(a, b):
        key = tuple(sorted((a, b)))
        if key not in mid_cache:
            mid_cache[key] = len(verts)
            verts.append([(verts[a][0]+verts[b][0])/2,
                          (verts[a][1]+verts[b][1])/2,
                          (verts[a][2]+verts[b][2])/2])
        return mid_cache[key]
    new_faces = []
    for (a, b, c) in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return verts, new_faces


def generate(data_dir, out_dir, tag):
    d = np.load(data_dir / "pred.npz")
    kpts2d = d["kpts2d"]
    preds3d = d["preds3d"]
    fps = float(d["fps"]) if "fps" in d else 30.0
    T = len(kpts2d)
    delay_ms = int(1000.0 / fps)

    SKEL = [(0,7),(7,8),(8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),
            (15,16),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6)]

    def skel_pts(X):
        xs, ys = [], []
        for (i, j) in SKEL:
            xs += [X[i,0], X[j,0], None]
            ys += [X[i,1], X[j,1], None]
        return xs, ys

    def skel_pts3_colored(X):
        xs, ys, zs, cols = [], [], [], []
        z = X[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        span = zmax - zmin + 1e-6
        for (i, j) in SKEL:
            xs += [X[i,0], X[j,0], None]
            ys += [-X[i,1], -X[j,1], None]
            zs += [X[i,2], X[j,2], None]
            cols += [(z[i]-zmin)/span, (z[j]-zmin)/span, None]
        return xs, ys, zs, cols

    # 球体网格 (半径 25mm)
    R = 25.0
    sphere_v, sphere_f = icosphere(R, subdiv=0)  # 12 顶点, 20 面 (小体积)
    NV = len(sphere_v)
    # 17 球全局面索引 (每球 NV 顶点, 面索引 = 球偏移 + 相对)
    base_i = []
    for j in range(17):
        for f in sphere_f:
            base_i += [int(f[0]) + j*NV, int(f[1]) + j*NV, int(f[2]) + j*NV]
    # 每帧 17 球: 顶点平铺 (17*NV, 3)
    def balls(t):
        P = preds3d[t]
        xs, ys, zs, mc = [], [], [], []
        z = P[:, 2]; zmin, zmax = float(z.min()), float(z.max()); span = zmax-zmin+1e-6
        for j in range(17):
            xs += list(P[j,0] + sphere_v[:,0])
            ys += list(-(P[j,1]) + sphere_v[:,1])
            zs += list(P[j,2] + sphere_v[:,2])
            mc += [(z[j]-zmin)/span] * NV
        return xs, ys, zs, mc

    f2, f3, fball = [], [], []
    for t in range(T):
        x2, y2 = skel_pts(kpts2d[t])
        x3, y3, z3, c3 = skel_pts3_colored(preds3d[t])
        bx, by, bz, bmc = balls(t)
        f2.append({"x": x2, "y": y2})
        f3.append({"x": x3, "y": y3, "z": z3, "c": c3})
        fball.append({"x": bx, "y": by, "z": bz, "mc": bmc})

    all2d = kpts2d.reshape(-1, 2)
    xmin, xmax = float(all2d[:,0].min())-20, float(all2d[:,0].max())+20
    ymin, ymax = float(all2d[:,1].min())-20, float(all2d[:,1].max())+20
    zmax = float(np.abs(preds3d).max()) + 50

    f2_j = json.dumps(_conv(f2))
    f3_j = json.dumps(_conv(f3))
    fball_j = json.dumps(_conv(fball))
    idx_j = json.dumps(base_i)

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%s</title>
<script src="plotly.min.js"></script>
<style>
body { font-family: Arial; margin: 20px; background: #1a1a2e; color: #eee; }
h2 { text-align: center; }
.container { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }
.panel { background: #16213e; border-radius: 10px; padding: 8px; }
#vimg { width: 500px; height: 440px; object-fit: contain; background: #000; border-radius: 8px; }
.controls { display: flex; justify-content: center; gap: 10px; margin: 15px 0; align-items: center; }
button { padding: 10px 24px; font-size: 16px; cursor: pointer; border-radius: 6px; border: none;
         background: #0f3460; color: #fff; }
button:hover { background: #16213e; }
input[type=range] { width: 500px; }
#fpsinfo { color: #aaa; }
</style></head><body>
<h2>%s</h2>
<div class="controls">
  <button onclick="playPause()">Play/Pause</button>
  <input type="range" id="slider" min="0" max="%d" value="0" oninput="seek(this.value)">
  <span id="fpsinfo">%d fps · 帧 <span id="fnum">0</span>/%d</span>
</div>
<div class="container">
  <div class="panel"><img id="vimg" src="frame_000.jpg"></div>
  <div class="panel"><div id="p2d"></div></div>
  <div class="panel"><div id="p3d"></div></div>
</div>
<script>
var T = %d; var delay = %d;
var f2 = %s; var f3 = %s; var fball = %s;
var FACE = %s; var NV = %d;
var playing = false; var cur = 0; var timer = null;

function render(t) {
  document.getElementById('vimg').src = 'frame_' + String(t).padStart(3,'0') + '.jpg';
  Plotly.restyle('p2d', {x: [f2[t].x], y: [f2[t].y]}, [0]);
  // 只更新数据, 不碰 camera/layout -> 不打断用户拖拽
  Plotly.restyle('p3d', {
    x: [f3[t].x], y: [f3[t].y], z: [f3[t].z], 'line.color': [f3[t].c]}, [0]);
  Plotly.restyle('p3d', {
    x: [fball[t].x], y: [fball[t].y], z: [fball[t].z],
    'vertexcolor': [fball[t].mc]}, [1]);
  document.getElementById('slider').value = t;
  document.getElementById('fnum').textContent = t;
}
function playPause() {
  if (playing) { playing = false; clearInterval(timer); }
  else { playing = true; timer = setInterval(function(){
    cur = (cur + 1) %% T; render(cur); }, delay); }
}
function seek(v) { cur = +v; render(cur); }

var colorscale = [[0, 'rgb(40,60,200)'], [0.5, 'rgb(220,220,80)'], [1, 'rgb(220,40,40)']];
var layout2 = {title: '2D 关键点', width: 500, height: 440,
  xaxis: {range: [%f, %f]}, yaxis: {range: [%f, %f], autorange: 'reversed'}};
var layout3 = {title: '3D 姿态 (光照球体, 可拖拽)', width: 500, height: 440,
  scene: {xaxis: {range: [-%f, %f]}, yaxis: {range: [-%f, %f]},
          zaxis: {range: [-%f, %f]}, aspectmode: 'cube',
          camera: {eye: {x: 0.0, y: 0.0, z: 2.5}, up: {x: 0, y: 1, z: 0}}}};

Plotly.newPlot('p2d', [{x: f2[0].x, y: f2[0].y, type: 'scatter', mode: 'lines+markers',
  line: {color: 'lime', width: 3}, marker: {size: 5, color: 'red'}}], layout2);
Plotly.newPlot('p3d', [
  {x: f3[0].x, y: f3[0].y, z: f3[0].z, type: 'scatter3d', mode: 'lines',
   line: {color: f3[0].c, width: 5, colorscale: colorscale, showscale: false}},
  {x: fball[0].x, y: fball[0].y, z: fball[0].z, type: 'mesh3d',
   i: FACE.filter(function(v, idx){ return idx %% 3 === 0; }),
   j: FACE.filter(function(v, idx){ return idx %% 3 === 1; }),
   k: FACE.filter(function(v, idx){ return idx %% 3 === 2; }),
   vertexcolor: fball[0].mc, colorscale: colorscale,
   lighting: {ambient: 0.4, diffuse: 0.8, specular: 0.6, roughness: 0.3,
              fresnel: 0.2},
   flatshading: false, opacity: 0.95}
], layout3);
</script></body></html>
""" % (tag, tag, T-1, int(fps), T, T, delay_ms, f2_j, f3_j, fball_j,
       idx_j, NV, xmin, xmax, ymin, ymax, zmax, zmax, zmax, zmax, zmax, zmax)

    out = Path(out_dir) / "inference.html"
    out.write_text(html)
    print("HTML 已生成:", out, "| 帧:", T)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default="Pose Inference")
    args = ap.parse_args()
    generate(Path(args.data), Path(args.data), args.tag)
