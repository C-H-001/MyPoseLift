from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from mypose.data.keypoints65 import COCO65_EDGES


def _device_from_config(cfg: dict) -> torch.device:
    requested = cfg["train"]["device"]
    selected = (
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    return torch.device(selected)


def _round_points(points: np.ndarray, digits: int = 3) -> list[list[float]]:
    return np.round(points.astype(np.float32), digits).tolist()


def _frame_mpjpe(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not np.any(valid):
        return float("nan")
    return float(np.linalg.norm(prediction[valid] - target[valid], axis=-1).mean())


def _select_sequence_indices(
    dataset: Any,
    num_sequences: int,
    frames_per_sequence: int,
    start_index: int,
) -> list[list[int]]:
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    if frames_per_sequence <= 0:
        raise ValueError(
            f"frames_per_sequence must be positive, got {frames_per_sequence}"
        )
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")

    sequences: list[list[int]] = []
    seen: set[str] = set()
    for index in range(start_index, len(dataset)):
        sequence_id = str(dataset.sequence_ids[index])
        if sequence_id in seen:
            continue
        seen.add(sequence_id)
        ordered = list(dataset._ordered_by_sequence.get(sequence_id, [index]))
        if not ordered:
            continue
        sequences.append(ordered[:frames_per_sequence])
        if len(sequences) >= num_sequences:
            break
    if not sequences:
        raise ValueError("no sequences available for export")
    return sequences


def _sequence_label(meta: dict[str, Any], fallback: str) -> str:
    subject = meta.get("subject")
    action = meta.get("action")
    camera = meta.get("camera")
    if subject not in (None, "") and action not in (None, ""):
        parts = [str(subject), str(action)]
        if camera not in (None, ""):
            parts.append(str(camera))
        return " / ".join(parts)
    return fallback


def _collect_payload(
    cfg: dict,
    checkpoint_path: Path,
    cache_path: Path,
    num_sequences: int,
    frames_per_sequence: int,
    start_index: int,
) -> dict[str, Any]:
    from mypose.data.h3wb import H3WBDataset
    from mypose.engine import build_model_from_config
    from mypose.engine.checkpoint import load_checkpoint

    device = _device_from_config(cfg)
    dataset = H3WBDataset(cache_path, window=int(cfg["data"]["window"]))
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    sequences = []
    with torch.no_grad():
        for indices in _select_sequence_indices(
            dataset, num_sequences, frames_per_sequence, start_index
        ):
            frames = []
            for index in indices:
                sample = dataset[index]
                history = (
                    torch.from_numpy(sample["history_2d"])
                    .unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                )
                prediction = model(history)[0].cpu().numpy()
                target = sample["target_3d"]
                mask = sample["target_mask"]
                meta = dict(sample["meta"])
                frames.append(
                    {
                        "frame": str(meta.get("frame_id", index)),
                        "mpjpe": round(_frame_mpjpe(prediction, target, mask), 3),
                        "input2d": _round_points(sample["history_2d"][-1]),
                        "gt3d": _round_points(target),
                        "pred3d": _round_points(prediction),
                    }
                )
            first_meta = dict(dataset[indices[0]]["meta"])
            sequence_id = str(first_meta.get("sequence_id", "sequence"))
            sequences.append(
                {
                    "id": sequence_id,
                    "label": _sequence_label(first_meta, sequence_id),
                    "frames": frames,
                }
            )
    return {
        "edges": COCO65_EDGES,
        "sequences": sequences,
    }


def _html_document(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3WB 65-point sequence viewer</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #101214;
  --fg: #eceff3;
  --muted: #9aa4b2;
  --panel: #181b20;
  --grid: #3a414b;
  --gt: #4ea1ff;
  --pred: #ff9c3a;
  --input: #72d572;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f7f8fa;
    --fg: #1d2430;
    --muted: #5c6675;
    --panel: #ffffff;
    --grid: #d7dce3;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 20px;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: var(--bg);
  color: var(--fg);
}}
main {{
  max-width: 1280px;
  margin: 0 auto;
}}
h1 {{
  margin: 0 0 14px;
  font-size: 22px;
  font-weight: 500;
}}
.controls {{
  display: flex;
  gap: 16px;
  align-items: end;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
label {{
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 14px;
}}
select, input[type="range"] {{
  min-width: 220px;
}}
.frameControl {{
  flex: 1 1 360px;
}}
.frameControl input {{
  width: 100%;
}}
.status {{
  color: var(--muted);
  min-width: 210px;
}}
.viewer {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
.pane {{
  background: var(--panel);
  border: 1px solid var(--grid);
  min-height: 420px;
  padding: 12px;
}}
.paneHeader {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}}
.title {{
  font-weight: 500;
}}
.legend {{
  display: flex;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
}}
.swatch {{
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 4px;
  border-radius: 50%;
}}
svg {{
  width: 100%;
  height: 560px;
  display: block;
}}
.edge {{
  fill: none;
  stroke-linecap: round;
}}
.joint {{
  stroke: none;
}}
@media (max-width: 800px) {{
  body {{ padding: 12px; }}
  .viewer {{ grid-template-columns: 1fr; }}
  svg {{ height: 420px; }}
}}
</style>
</head>
<body>
<main>
<h1>H3WB 65-point sequence viewer</h1>
<div class="controls">
  <label>Sequence
    <select id="sequenceSelect"></select>
  </label>
  <label class="frameControl">Frame
    <input id="frameSlider" type="range" min="0" max="0" value="0">
  </label>
  <div class="status" id="frameStatus"></div>
</div>
<div class="viewer">
  <section class="pane">
    <div class="paneHeader">
      <div class="title">2D input</div>
      <div class="legend"><span><i class="swatch" style="background:var(--input)"></i>Input</span></div>
    </div>
    <svg id="view2d" role="img" aria-label="2D skeleton for selected frame"></svg>
  </section>
  <section class="pane">
    <div class="paneHeader">
      <div class="title">3D ground truth and prediction</div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--gt)"></i>Ground truth</span>
        <span><i class="swatch" style="background:var(--pred)"></i>Prediction</span>
      </div>
    </div>
    <svg id="view3d" role="img" aria-label="3D ground truth and prediction for selected frame"></svg>
  </section>
</div>
</main>
<script>
const sequenceViewerData = {data};
const edges = sequenceViewerData.edges;
const seqSelect = document.getElementById("sequenceSelect");
const slider = document.getElementById("frameSlider");
const statusEl = document.getElementById("frameStatus");
const svg2d = document.getElementById("view2d");
const svg3d = document.getElementById("view3d");

function project3d(p) {{
  return [p[0] + p[2] * 0.35, -p[1] + p[2] * 0.18];
}}

function bounds(pointSets) {{
  const pts = pointSets.flat();
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxX - minX < 1e-6) {{ minX -= 1; maxX += 1; }}
  if (maxY - minY < 1e-6) {{ minY -= 1; maxY += 1; }}
  return {{minX, maxX, minY, maxY}};
}}

function scalePoint(p, box, width, height) {{
  const pad = 28;
  const sx = (width - pad * 2) / (box.maxX - box.minX);
  const sy = (height - pad * 2) / (box.maxY - box.minY);
  const s = Math.min(sx, sy);
  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  return [
    width / 2 + (p[0] - cx) * s,
    height / 2 + (p[1] - cy) * s
  ];
}}

function clear(svg) {{
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}}

function line(svg, a, b, color, opacity) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
  el.setAttribute("x1", a[0]);
  el.setAttribute("y1", a[1]);
  el.setAttribute("x2", b[0]);
  el.setAttribute("y2", b[1]);
  el.setAttribute("stroke", color);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("opacity", opacity);
  el.setAttribute("class", "edge");
  svg.appendChild(el);
}}

function dot(svg, p, color, r) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  el.setAttribute("cx", p[0]);
  el.setAttribute("cy", p[1]);
  el.setAttribute("r", r);
  el.setAttribute("fill", color);
  el.setAttribute("class", "joint");
  svg.appendChild(el);
}}

function drawPose(svg, pointSets) {{
  clear(svg);
  const width = svg.clientWidth || 600;
  const height = svg.clientHeight || 420;
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  const box = bounds(pointSets.map(s => s.points));
  for (const set of pointSets) {{
    const scaled = set.points.map(p => scalePoint(p, box, width, height));
    for (const [a, b] of edges) {{
      line(svg, scaled[a], scaled[b], set.color, set.opacity);
    }}
    for (const p of scaled) {{
      dot(svg, p, set.color, set.radius);
    }}
  }}
}}

function activeFrame() {{
  const sequence = sequenceViewerData.sequences[seqSelect.selectedIndex];
  return {{
    sequence,
    frame: sequence.frames[Number(slider.value)]
  }};
}}

function render() {{
  const {{sequence, frame}} = activeFrame();
  const input2d = frame.input2d.map(p => [p[0], -p[1]]);
  const gt = frame.gt3d.map(project3d);
  const pred = frame.pred3d.map(project3d);
  drawPose(svg2d, [{{points: input2d, color: "var(--input)", opacity: 0.85, radius: 2.8}}]);
  drawPose(svg3d, [
    {{points: gt, color: "var(--gt)", opacity: 0.72, radius: 2.6}},
    {{points: pred, color: "var(--pred)", opacity: 0.72, radius: 2.6}}
  ]);
  statusEl.textContent = `Frame ${{Number(slider.value) + 1}}/${{sequence.frames.length}} | id ${{frame.frame}} | MPJPE ${{frame.mpjpe.toFixed(1)}} mm`;
}}

function setSequence(index) {{
  const sequence = sequenceViewerData.sequences[index];
  slider.max = Math.max(0, sequence.frames.length - 1);
  slider.value = 0;
  render();
}}

for (const sequence of sequenceViewerData.sequences) {{
  const option = document.createElement("option");
  option.value = sequence.id;
  option.textContent = sequence.label;
  seqSelect.appendChild(option);
}}
seqSelect.addEventListener("change", () => setSequence(seqSelect.selectedIndex));
slider.addEventListener("input", render);
window.addEventListener("resize", render);
setSequence(0);
</script>
</body>
</html>
"""


def export_sequence_viewer(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    cache_path: Path | None = None,
    num_sequences: int = 3,
    frames_per_sequence: int = 120,
    start_index: int = 0,
) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache = cache_path or Path(cfg["data"]["val_cache"])
    payload = _collect_payload(
        cfg,
        checkpoint_path,
        cache,
        num_sequences,
        frames_per_sequence,
        start_index,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(payload), encoding="utf-8")
    print(f"wrote {html.escape(str(output_path))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an interactive H3WB 65-point sequence HTML viewer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--frames-per-sequence", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/h3wb_sequence_viewer.html"))
    args = parser.parse_args()
    export_sequence_viewer(
        args.config,
        args.checkpoint,
        args.out,
        cache_path=args.cache,
        num_sequences=args.num_sequences,
        frames_per_sequence=args.frames_per_sequence,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
