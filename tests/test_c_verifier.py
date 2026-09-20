"""Read-only C verifier cross-checks; no search or production archive writes."""

import itertools
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from solver.correlation import full_correlation_profile
from solver.verifier import verify_pqcp


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler not available")
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path_factory.mktemp("c_verifier") / "verify_pqcp"
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
         str(root / "verify_pqcp.c"), "-o", str(binary)], check=True,
        capture_output=True, text=True,
    )
    return binary


def record(a="0101", b="0011"):
    profile = full_correlation_profile(a, b)
    shifts = [u for u in range(1, len(a)) if profile[u]]
    # Invalid pairs still receive parseable metadata; reference verifier decides validity.
    stored_shifts = shifts if len(shifts) == 2 else [1, 2]
    pacs = profile[shifts[0]] if shifts else 4
    return ("L={}\nnonzero shifts={},{}\nnonzero PACS={}\na={}\nb={}\n\n"
            .format(len(a), *stored_shifts, pacs, a, b))


def run(executable, tmp_path, contents, length=4, interactive=False):
    path = tmp_path / "{}.txt".format(length)
    path.write_bytes(contents.encode())
    before = path.read_bytes()
    result = subprocess.run(
        [str(executable)] if interactive else [str(executable), str(length)],
        cwd=tmp_path, input="{}\n".format(length), capture_output=True,
        text=True, timeout=15,
    )
    assert path.read_bytes() == before
    return result


def test_interactive_and_line_endings(executable, tmp_path):
    result = run(executable, tmp_path, record().replace("\n", "\r\n").rstrip(),
                 interactive=True)
    assert result.returncode == 0
    assert "通過：1" in result.stdout


@pytest.mark.parametrize("length", range(1, 5))
def test_all_small_pairs_match_python_verifier(executable, tmp_path, length):
    sequences = ["".join(bits) for bits in itertools.product("01", repeat=length)]
    pairs = list(itertools.product(sequences, repeat=2))
    expected = sum(verify_pqcp(a, b).is_valid for a, b in pairs)
    result = run(executable, tmp_path, "".join(record(a, b) for a, b in pairs), length)
    assert result.returncode == (0 if expected == len(pairs) else 1)
    assert "總紀錄：{}\n通過：{}\n".format(len(pairs), expected) in result.stdout


@pytest.mark.parametrize("length", [5, 8, 16, 44, 46, 68, 94])
def test_random_pairs_match_python_verifier(executable, tmp_path, length):
    rng = random.Random(length)
    pairs = [("".join(rng.choice("01") for _ in range(length)),
              "".join(rng.choice("01") for _ in range(length))) for _ in range(30)]
    expected = sum(verify_pqcp(a, b).is_valid for a, b in pairs)
    result = run(executable, tmp_path, "".join(record(a, b) for a, b in pairs), length)
    assert "總紀錄：30\n通過：{}\n".format(expected) in result.stdout


def test_existing_known_44_pair(executable, tmp_path):
    a = "11011001010100000010001100011110000100110111"
    b = "01101000110010000111101000100011010010010100"
    assert verify_pqcp(a, b).is_valid
    assert run(executable, tmp_path, record(a, b), 44).returncode == 0


@pytest.mark.parametrize("old,new", [
    ("L=4", "L=6"),
    ("shifts=1,3", "shifts=1,2"),
    ("PACS=-4", "PACS=4"),
    ("a=0101", "a=010x"),
    ("b=0011", "b=001"),
    ("b=0011", ""),
    ("a=0101", "a=0101\na=0101"),
    ("L=4", "L=9999999999999999999999999999999999999999"),
    ("shifts=1,3", "shifts=1,3,2"),
    ("b=0011", "b=0011\0junk"),
    ("a=0101", "unexpected\na=0101"),
])
def test_rejects_malformed_or_inconsistent_record(executable, tmp_path, old, new):
    assert old in record()
    result = run(executable, tmp_path, record().replace(old, new))
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_does_not_stop_at_invalid_record(executable, tmp_path):
    result = run(executable, tmp_path, record() + record("0000", "0000") + record())
    assert result.returncode == 1
    assert "總紀錄：3\n通過：2\n不通過（含格式錯誤）：1" in result.stdout


def test_empty_file_fails(executable, tmp_path):
    assert run(executable, tmp_path, "\n\n").returncode == 1


@pytest.mark.parametrize("length", ["0", "-1", "4x", "999999999999999999999999"])
def test_bad_length(executable, tmp_path, length):
    result = subprocess.run([str(executable), length], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2


def test_missing_file(executable, tmp_path):
    result = subprocess.run([str(executable), "44"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2


def test_explicit_file_path(executable, tmp_path):
    path = tmp_path / "custom.txt"
    path.write_text(record())
    result = subprocess.run([str(executable), "4", str(path)], capture_output=True)
    assert result.returncode == 0
