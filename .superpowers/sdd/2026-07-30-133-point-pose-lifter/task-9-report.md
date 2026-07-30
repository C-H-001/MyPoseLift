# Task 9 Report

## Documentation

- Created `README.md` with Windows virtual-environment setup, editable install,
  dataset download and H3WB preparation, both training configurations, sample
  inspection, streaming inference guidance, and pytest verification.
- Updated `docs/datasets.md` with the H3WB preparation command, the COCO cache
  clarification, and current cache/output guidance.
- No dataset files were added or modified.

## Verification

### Full test suite

Command:

```text
pytest -v
```

Result: blocked during collection. Pytest collected 25 items and reported 6
collection errors because importing PyTorch failed with:

```text
OSError: [WinError 1114] DLL initialization routine failed. Error loading "E:\Anaconda\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

The affected modules were `test_causal_window.py`, `test_engine_smoke.py`,
`test_h3wb.py`, `test_infer_stream.py`, `test_metrics_losses.py`, and
`test_model_forward.py`.

### Import smoke test

Command:

```text
python -c "from mypose.models.hrgcn_lifter import HRGCNLifter; HRGCNLifter(hidden_channels=16); print('ok')"
```

Result: blocked by the same `OSError: [WinError 1114]` while importing
`torch` and loading `c10.dll`; `ok` was not printed.

### CPU forward smoke test

Command:

```text
python -c "import torch; from mypose.models.hrgcn_lifter import HRGCNLifter; m=HRGCNLifter(hidden_channels=16,use_temporal=True); y=m(torch.zeros(1,3,133,3)); print(tuple(y.shape))"
```

Result: blocked by the same `OSError: [WinError 1114]` while importing
`torch`; the shape was not printed.

### Non-Torch syntax check

Command:

```text
python -m compileall -q mypose tools tests
```

Result: exit code 0.

### Dataset staging check

`git status --short` showed only the requested documentation changes before the
commit, and no files under `data/raw`, `data/processed`, or `data/cache` were
staged or tracked.

### Diff check

`git diff --check` and `git diff --cached --check` completed without whitespace
errors.

## Commit

```text
210af9e docs: add pose lifter workflow
```

## Self-review

The README includes the exact setup, download, preparation, training,
inspection, and test commands from the Task 9 brief. It documents editable
installation as the intended import-path setup and keeps the Windows venv
syntax. The repository changes are limited to `README.md` and
`docs/datasets.md`; no implementation code or dataset artifacts were changed.

The only unresolved verification concern is the environment's native PyTorch
`c10.dll` loading failure. A clean rerun in a working PyTorch environment is
required for the six Torch-dependent test modules and both Torch smoke tests.

## Review Fixes

The baseline was subsequently verified in the working
`E:\Anaconda\envs\detection_demo` environment:

```text
& 'E:\Anaconda\envs\detection_demo\python.exe' -m pytest -v
```

Result: 50 passed in 11.40s.

```text
& 'E:\Anaconda\envs\detection_demo\python.exe' -c "from mypose.models.hrgcn_lifter import HRGCNLifter; HRGCNLifter(hidden_channels=16); print('import_smoke_ok')"
```

Result: `import_smoke_ok`.

```text
& 'E:\Anaconda\envs\detection_demo\python.exe' -c "import torch; from mypose.models.hrgcn_lifter import HRGCNLifter; m=HRGCNLifter(hidden_channels=16,use_temporal=True); y=m(torch.zeros(1,3,133,3)); print(tuple(y.shape))"
```

Result: `(1, 133, 3)`.

The default `E:\Anaconda` environment still has the environment-specific
PyTorch `torch/c10.dll` initialization failure described above; it is a caveat,
not the baseline result.

README now includes a standalone `Evaluate` procedure using
`mypose.engine.evaluate` with `--config` and `--checkpoint`. The module now
provides that CLI while preserving the existing `evaluate()` API.
