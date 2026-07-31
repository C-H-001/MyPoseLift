# Sequence HTML Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static HTML sequence viewer that scrubs H3WB 65-point validation sequences with 2D input and overlaid 3D GT/Pred.

**Architecture:** A single export CLI loads the existing dataset/model/checkpoint, selects contiguous sequence frames, runs predictions, and writes self-contained HTML with embedded compact JSON. Tests use a synthetic cache and checkpoint so the export path is verified without the large H3WB release.

**Tech Stack:** Python, PyTorch, NumPy, PyYAML, pytest, static HTML/SVG/JavaScript.

## Global Constraints

- Use the active 65-point layout and `COCO65_EDGES`.
- Default export is first 3 validation sequences, up to 120 frames each.
- The HTML must have a slider for frame scrubbing.
- The left pane shows 2D skeleton input.
- The right pane overlays 3D ground truth and prediction.
- The exported HTML is a generated report and must not be committed.
- No web server or external CDN dependency is required.

---

### Task 1: Export CLI and Tests

**Files:**
- Create: `tools/export_sequence_viewer.py`
- Create: `tests/test_export_sequence_viewer.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `H3WBDataset(cache_file, window)`, `build_model_from_config(cfg)`, `load_checkpoint(path, model)`, `COCO65_EDGES`.
- Produces: `export_sequence_viewer(config_path, checkpoint_path, output_path, cache_path=None, num_sequences=3, frames_per_sequence=120, start_index=0) -> None`.

- [ ] **Step 1: Write failing CLI/export tests**

Add `tests/test_export_sequence_viewer.py` with:

```python
def test_export_sequence_viewer_help_runs():
    result = subprocess.run(
        [sys.executable, "tools/export_sequence_viewer.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--frames-per-sequence" in result.stdout


def test_export_sequence_viewer_writes_slider_html(tmp_path):
    cache, checkpoint, config = make_tiny_cache_checkpoint_and_config(tmp_path)
    out = tmp_path / "viewer.html"
    subprocess.run(
        [
            sys.executable,
            "tools/export_sequence_viewer.py",
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--cache", str(cache),
            "--num-sequences", "1",
            "--frames-per-sequence", "2",
            "--out", str(out),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    html = out.read_text(encoding="utf-8")
    assert 'type="range"' in html
    assert "Ground truth" in html
    assert "Prediction" in html
    assert "sequenceViewerData" in html
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
& 'E:\Anaconda\envs\detection_demo\python.exe' -m pytest tests\test_export_sequence_viewer.py -q
```

Expected: FAIL because `tools/export_sequence_viewer.py` does not exist.

- [ ] **Step 3: Implement export CLI**

Create `tools/export_sequence_viewer.py` with:

- `export_sequence_viewer(...)`
- `_select_sequence_indices(dataset, num_sequences, frames_per_sequence, start_index)`
- `_frame_mpjpe(prediction, target, mask)`
- `_html_document(payload)`
- `main()`

The HTML must use native `<select>` and `<input type="range">`, two SVG panes, and local JavaScript only.

- [ ] **Step 4: Update README**

Add:

```powershell
python tools/export_sequence_viewer.py --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt --out reports/h3wb_sequence_viewer.html
```

- [ ] **Step 5: Run tests**

Run:

```powershell
& 'E:\Anaconda\envs\detection_demo\python.exe' -m pytest tests\test_export_sequence_viewer.py tests\test_plot_prediction.py -q
& 'E:\Anaconda\envs\detection_demo\python.exe' -m pytest -q
```

Expected: PASS.

- [ ] **Step 6: Generate real viewer**

Run:

```powershell
& 'E:\Anaconda\envs\detection_demo\python.exe' tools\export_sequence_viewer.py --config configs\h3wb_tcn_t81.yaml --checkpoint checkpoints\h3wb_tcn_t81\best.pt --out reports\h3wb_sequence_viewer.html
```

Expected: writes a non-empty HTML file.

- [ ] **Step 7: Commit and push**

```powershell
git add docs/superpowers/specs/2026-08-01-sequence-html-viewer-design.md docs/superpowers/plans/2026-08-01-sequence-html-viewer.md tools/export_sequence_viewer.py tests/test_export_sequence_viewer.py README.md
git commit -m "feat: add sequence HTML viewer export"
git push
```
