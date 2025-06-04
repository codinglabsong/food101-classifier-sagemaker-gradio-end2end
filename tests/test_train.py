import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import ensure_unpacked


def test_ensure_unpacked_noop(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "file.txt").write_text("hi")
    assert ensure_unpacked(str(d)) == str(d)
