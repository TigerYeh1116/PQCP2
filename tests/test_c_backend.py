"""Correctness and integration tests for the compiled C search backend."""

from pathlib import Path
import pytest

from solver.c_backend import (
    CSearchConfig,
    ensure_c_backend,
    ensure_compressed_fkm_seed_bank,
    run_c_search,
)
from solver.target_profiles import pair_content, target_content_profiles
from solver.verifier import verify_pqcp


def test_compressed_seed_bank_covers_every_required_a_content(tmp_path):
    path = ensure_compressed_fkm_seed_bank(44, 123, tmp_path, seeds_per_content=4)
    words = tuple(
        tuple(int(bit) for bit in line)
        for line in path.read_text(encoding="ascii").splitlines()
    )
    assert words and all(len(word) == 44 and set(word) <= {0, 1} for word in words)
    observed = {pair_content(word, (0,) * 44)[:2] for word in words}
    required = {
        (profile.a_even_ones, profile.a_odd_ones)
        for profile in target_content_profiles(44, decimation_reduced=True)
    }
    assert observed == required


def test_seed_bank_reuse_requires_intact_metadata_and_content(tmp_path):
    first = ensure_compressed_fkm_seed_bank(44, 9, tmp_path, seeds_per_content=2)
    original = first.read_text(encoding="ascii")
    first.write_text("broken\n", encoding="ascii")
    second = ensure_compressed_fkm_seed_bank(44, 9, tmp_path, seeds_per_content=2)
    assert second == first
    assert second.read_text(encoding="ascii") == original


@pytest.mark.parametrize("direct_sampling", [False, True])
def test_c_backend_compiles_and_reports_exact_counters(tmp_path, direct_sampling):
    executable = ensure_c_backend(executable=tmp_path / "pqcp_search_c")
    assert executable.exists()
    result = run_c_search(CSearchConfig(
        L=44,
        seed=321,
        seconds=0.02,
        threads=1,
        seeds_per_content=2,
        root=tmp_path,
        executable=executable,
        echo=False,
        direct_swap_sampling=direct_sampling,
    ))
    assert result.restarts > 0
    assert result.moves > 0
    assert result.swap_evaluations == 2 * result.moves
    assert result.fkm_seeds_applied > 0
    assert result.fkm_seed_misses == 0
    assert result.log_path.exists()
    assert "compressed FKM seed bank enabled" in result.log_path.read_text(encoding="utf-8")


def test_every_c_candidate_is_independently_verified_before_write(tmp_path):
    result = run_c_search(CSearchConfig(
        L=4,
        seed=17,
        seconds=0.001,
        threads=1,
        seeds_per_content=2,
        root=tmp_path,
        executable=tmp_path / "pqcp_search_c",
        echo=False,
    ))
    assert result.valid_results == result.verified_candidates
    assert result.verified_candidates > 0
    records = (tmp_path / "4.txt").read_text(encoding="utf-8").splitlines()
    sequences = [line[2:] for line in records if line.startswith(("a=", "b="))]
    for index in range(0, len(sequences), 2):
        a = tuple(map(int, sequences[index]))
        b = tuple(map(int, sequences[index + 1]))
        assert verify_pqcp(a, b).is_valid


def test_display_counts_persisted_pairs_not_raw_or_stale_hits(tmp_path, monkeypatch):
    import sys
    import solver.c_backend as backend

    candidate = "PQCP_CANDIDATE L=4 k=1 sign=1 energy=0 a={} b={}"
    stats = "搜尋統計：restart=3, moves=10, swap evaluations=20, valid results={}"
    lines = [
        candidate.format("0000", "0011"),
        stats.format(0),  # stale snapshot must not hide the newly saved pair
        candidate.format("0011", "0000"),  # A/B exchange is a duplicate
        stats.format(99),  # raw hits must not inflate the displayed count
        candidate.format("0000", "0110"),
        stats.format(3),
    ]
    popen = backend.subprocess.Popen
    monkeypatch.setattr(backend, "ensure_c_backend", lambda *_args: tmp_path / "fake")
    monkeypatch.setattr(backend, "ensure_compressed_fkm_seed_bank", lambda *_args: tmp_path / "seeds")
    monkeypatch.setattr(backend.subprocess, "Popen", lambda _cmd, **kwargs: popen(
        [sys.executable, "-u", "-c", "print({!r}, flush=True)".format("\n".join(lines))],
        **kwargs,
    ))
    output = []
    result = run_c_search(CSearchConfig(L=4, root=tmp_path, echo=False), line_callback=output.append)
    displayed = [int(line.rsplit("=", 1)[1]) for line in output if line.startswith("搜尋統計：")]
    assert displayed == [1, 1, 1, 2, 2]
    assert result.valid_results == result.verified_candidates == 3
    assert result.new_solutions == 2
    assert (tmp_path / "4.txt").read_text().count("\na=") == 2
    assert "valid results=99" in result.log_path.read_text()  # raw diagnostic retained
