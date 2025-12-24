import os
import subprocess
import sys
from pathlib import Path

import pytest


SNAPSHOT_DIR = Path("tests/snapshots")
FIXTURES = [
    ("x", Path("tests/fixtures/x.ptx")),
    ("saxpy", Path("tests/fixtures/saxpy.ptx")),
]


@pytest.mark.parametrize("name,fixture_path", FIXTURES)
@pytest.mark.xfail(reason="Snapshots pending update for new lowering")
def test_end_to_end_snapshot(name: str, fixture_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "ptx2cu.py", str(fixture_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ptx2cu failed: {result.stderr}"

    out = result.stdout
    snap_path = SNAPSHOT_DIR / f"{name}.cu"

    if os.getenv("UPDATE_SNAPSHOTS") == "1":
        snap_path.write_text(out)

    assert (
        snap_path.exists()
    ), f"Missing snapshot {snap_path} (run with UPDATE_SNAPSHOTS=1 to create)"
    expected = snap_path.read_text()
    assert out == expected
