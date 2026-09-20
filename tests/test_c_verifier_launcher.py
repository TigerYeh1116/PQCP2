"""Exercise the click-to-run entry without touching project result archives."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def project(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("C compiler not available")
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project with spaces"
    project.mkdir()
    for name in ("verify_pqcp.py", "verify_pqcp.c"):
        shutil.copyfile(root / name, project / name)
    (project / "4.txt").write_text(
        "L=4\nnonzero shifts=1,3\nnonzero PACS=-4\na=0101\nb=0011\n"
    )
    return project


def launch(project, cwd, length="4"):
    return subprocess.run(
        [sys.executable, str(project / "verify_pqcp.py")],
        cwd=cwd, input=length + "\n", text=True, capture_output=True, timeout=30,
    )


def test_click_entry_from_different_directory(project, tmp_path):
    original = (project / "4.txt").read_bytes()
    result = launch(project, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "通過：1" in result.stdout
    assert (project / "build/verify_pqcp").is_file()
    assert (project / "4.txt").read_bytes() == original
    # Existing build is replaced safely on the next launch too.
    assert launch(project, tmp_path).returncode == 0


def test_launcher_preserves_invalid_record_status(project, tmp_path):
    path = project / "4.txt"
    path.write_text(path.read_text().replace("a=0101", "a=0000"))
    result = launch(project, tmp_path)
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_failed_compile_keeps_binary_but_does_not_run_it(project, tmp_path):
    assert launch(project, tmp_path).returncode == 0
    executable = project / "build/verify_pqcp"
    original = executable.read_bytes()
    (project / "verify_pqcp.c").write_text("invalid C source\n")
    result = launch(project, tmp_path)
    assert result.returncode == 2
    assert "編譯失敗" in result.stderr
    assert "通過：" not in result.stdout
    assert executable.read_bytes() == original
    assert list((project / "build").iterdir()) == [executable]
