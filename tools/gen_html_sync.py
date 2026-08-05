"""test.mp4 完整视频: 同步 HTML, 2D 图像坐标(y向下,x在下), 3D 正面"""
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


data_dir = Path("/home/user/ch/MyPoseLift/outputs/demo/rtmw_test")
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

def skel_pts3(X):
    xs, ys, zs = [], [], []
    for (i, j) in SKEL:
        xs += [X[i,0], X[j,0], None]
        ys += [-X[i,1], -X[j,1], None]   # y 翻转: 头部朝上
        zs += [X[i,2], X[j,2], None]
    return xs, ys, zs

f2, f3 = [], []
for t in range(T):
    x2, y2 = skel_pts(kpts2d[t])
    x3, y3, z3 = skel_pts3(preds3d[t])
    f2.append({"x": x2, "y": y2})
    f3.append({"x": x3, "y": y3, "z": z3})

all2d = kpts2d.reshape(-1, 2)
xmin, xmax = float(all2d[:,0].min())-20, float(all2d[:,0].max())+20
ymin, ymax = float(all2d[:,1].min())-20, float(all2d[:,1].max())+20
zmax = float(np.abs(preds3d).max()) + 50

f2_j, f3_j = json.dumps(_conv(f2)), json.dumps(_conv(f3))

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Test Pose Inference Sync</title>
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
<h2>完整视频推理 (test.mp4): RTMW 133点 → H36M 17点 → TCN 3D</h2>
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
var f2 = %s; var f3 = %s;
var playing = false; var cur = 0; var timer = null;

function render(t) {
  document.getElementById('vimg').src = 'frame_' + String(t).padStart(3,'0') + '.jpg';
  Plotly.restyle('p2d', {x: [f2[t].x], y: [f2[t].y]}, [0]);
  // 保存用户当前 3D 视角, 数据更新后恢复 (防止 camera 重置)
  var gd3 = document.getElementById('p3d');
  var cam = null;
  try { cam = gd3._fullLayout.scene.camera; } catch(e) {}
  Plotly.restyle('p3d', {x: [f3[t].x], y: [f3[t].y], z: [f3[t].z]}, [0]);
  if (cam) { Plotly.relayout('p3d', {'scene.camera': cam}); }
  document.getElementById('slider').value = t;
  document.getElementById('fnum').textContent = t;
}
function playPause() {
  if (playing) { playing = false; clearInterval(timer); }
  else { playing = true; timer = setInterval(function(){
    cur = (cur + 1) %% T; render(cur); }, delay); }
}
function seek(v) { cur = +v; render(cur); }

// 2D: 图像坐标 (y 向下, x 轴在底部) -> yaxis autorange reversed
var layout2 = {title: '2D 关键点 (图像坐标, x下y上)', width: 500, height: 440,
  xaxis: {range: [%f, %f], title: 'x'}, yaxis: {range: [%f, %f], autorange: 'reversed', title: 'y'}};
var layout3 = {title: '3D 姿态预测 (正面, 可旋转)', width: 500, height: 440,
  scene: {xaxis: {range: [-%f, %f], title: 'X'}, yaxis: {range: [-%f, %f], title: 'Y'},
          zaxis: {range: [-%f, %f], title: 'Z'}, aspectmode: 'cube',
          camera: {eye: {x: 0.0, y: 0.0, z: 2.5}, up: {x: 0, y: 1, z: 0}}}};

Plotly.newPlot('p2d', [{x: f2[0].x, y: f2[0].y, type: 'scatter', mode: 'lines+markers',
  line: {color: 'lime', width: 3}, marker: {size: 5, color: 'red'}}], layout2);
Plotly.newPlot('p3d', [{x: f3[0].x, y: f3[0].y, z: f3[0].z, type: 'scatter3d',
  mode: 'lines+markers', line: {color: 'cyan', width: 5}, marker: {size: 4, color: 'red'}}], layout3);
</script></body></html>
""" % (T-1, int(fps), T, T, delay_ms, f2_j, f3_j,
       xmin, xmax, ymin, ymax, zmax, zmax, zmax, zmax, zmax, zmax)

out = Path("/home/user/ch/MyPoseLift/outputs/demo/rtmw_test/inference.html")
out.write_text(html)
print("test 同步 HTML 已生成:", out, "| 帧:", T)
