import subprocess
import sys
from pathlib import Path


def test_inspect_sample_help_works_outside_repository_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "inspect_sample.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--cache" in result.stdout
