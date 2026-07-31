import subprocess
import sys


def test_plot_prediction_help_runs():
    result = subprocess.run(
        [sys.executable, "tools/plot_prediction.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
