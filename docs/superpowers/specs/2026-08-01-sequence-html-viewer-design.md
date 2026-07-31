# Sequence HTML Viewer Design

## Goal

Create a static HTML viewer for H3WB 65-point validation sequences. The viewer lets a user scrub through frames with a progress slider, showing the current 2D skeleton on the left and 3D ground truth plus prediction overlaid on the right.

## Scope

- Add a repository tool `tools/export_sequence_viewer.py`.
- Export one standalone HTML file, defaulting to `reports/h3wb_sequence_viewer.html`.
- Use existing configs, checkpoints, 65-point edges, `H3WBDataset`, and model loading.
- Default export: first 3 validation sequences, up to 120 frames each.
- Keep exported HTML and generated reports untracked.

## Interface

Command:

```powershell
python tools/export_sequence_viewer.py --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt --out reports/h3wb_sequence_viewer.html
```

Optional arguments:

- `--cache PATH`: override `cfg["data"]["val_cache"]`.
- `--num-sequences INT`: default `3`.
- `--frames-per-sequence INT`: default `120`.
- `--start-index INT`: default `0`.

## Data Flow

1. Load config and validation cache.
2. Build the model and load the checkpoint.
3. Iterate dataset indices in cache order and group by `meta["sequence_id"]`.
4. Select the requested number of sequences, starting at `--start-index`.
5. For each selected frame:
   - Store current-frame 2D skeleton from `history_2d[-1]`.
   - Store 3D ground truth from `target_3d`.
   - Run the model on `history_2d` and store 3D prediction.
   - Store per-frame MPJPE over valid target points.
6. Serialize compact rounded JSON into the HTML.

## Viewer Behavior

- A sequence selector switches between exported sequences.
- A range input scrubs frames.
- Left SVG renders the 2D skeleton.
- Right SVG renders an orthographic 3D projection with GT and Pred overlaid.
- The current frame label and MPJPE update when the slider moves.
- No network, server, or external JavaScript dependencies are required.

## Validation

- Unit tests cover CLI help, HTML export with a tiny synthetic cache/checkpoint, and presence of expected controls/data labels.
- Full pytest suite must pass.
