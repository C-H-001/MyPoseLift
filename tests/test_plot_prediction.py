import subprocess
import sys
import tomllib
from pathlib import Path


def test_plot_prediction_help_runs():
    result = subprocess.run(
        [sys.executable, "tools/plot_prediction.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout


def test_plotting_dependency_is_declared():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert any(dependency.startswith("matplotlib") for dependency in project["project"]["dependencies"])
