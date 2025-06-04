import tarfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import ensure_unpacked


# ---------- helpers ----------
def _make_tar(src_dir: Path, tar_path: Path):
    """
    Create a .tar.gz archive from all files inside src_dir.
    The archive will contain files relative to src_dir.
    """
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in src_dir.rglob("*"):
            tar.add(p, arcname=p.relative_to(src_dir))


def _create_example_files(base: Path, names=("a.txt", "b/c.txt")) -> None:
    """
    Populate base with a few small files to verify extraction results.
    """
    for name in names:
        full = base / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(f"content of {name}")


# ---------- 1. path is the tarball ----------
def test_tarball_path_is_unpacked(tmp_path: Path):
    # Arrange: build a tar.gz in tmp_path
    src = tmp_path / "src"
    _create_example_files(src)
    tarball = tmp_path / "data.tgz"
    _make_tar(src, tarball)

    # Act
    out_path = ensure_unpacked(str(tarball))

    # Assert
    out_path = Path(out_path)
    assert out_path.is_dir()
    # the temp dir should live alongside the tarball
    assert out_path.parent == tmp_path
    # files extracted and intact
    assert (out_path / "a.txt").read_text() == "content of a.txt"
    assert (out_path / "b" / "c.txt").read_text() == "content of b/c.txt"


# ---------- 2. directory containing exactly one tarball ----------
def test_directory_with_single_tarball(tmp_path: Path):
    # Arrange: dir/only_one.tgz
    work_dir = tmp_path / "data"
    work_dir.mkdir()
    src = tmp_path / "src_files"
    _create_example_files(src)
    tarball = work_dir / "only_one.tgz"
    _make_tar(src, tarball)

    # Act
    out_path = ensure_unpacked(str(work_dir))

    # Assert: same directory returned
    assert out_path == str(work_dir)
    # tarball should be gone
    assert not tarball.exists()
    # files should be unpacked in-place
    assert (work_dir / "a.txt").is_file()
    assert (work_dir / "b" / "c.txt").is_file()


# ---------- 3. directory with no tarball (noop) ----------
def test_directory_noop(tmp_path: Path):
    data_dir = tmp_path / "plain_dir"
    _create_example_files(data_dir)

    before = sorted(p.relative_to(data_dir) for p in data_dir.rglob("*"))
    out_path = ensure_unpacked(str(data_dir))
    after = sorted(p.relative_to(data_dir) for p in data_dir.rglob("*"))

    # unchanged path & contents
    assert out_path == str(data_dir)
    assert before == after


# ---------- 4. ordinary file (noop) ----------
def test_non_tar_file_noop(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("hello")

    out = ensure_unpacked(str(f))
    assert out == str(f)
    assert f.read_text() == "hello"
