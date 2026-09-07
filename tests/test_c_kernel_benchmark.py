"""Protect discovery metrics against censoring and double-counted solutions."""

import pytest
import sys

from experiments.benchmark_c_kernel_search import (
    summarize, save_verified_discoveries, trial, pair_key,
)


def test_censored_runs_and_swapped_duplicate_pairs_are_not_hidden():
    common = {"L": 4, "method": "pool", "elapsed": 60, "moves": 600}
    rows = [
        {**common, "hits": [{"A": "0000", "B": "0011"}],
         "new_solutions": 1, "first_new_seconds": 5, "censored": False},
        {**common, "hits": [{"A": "0011", "B": "0000"}],
         "new_solutions": 1, "first_new_seconds": 8, "censored": False},
        {**common, "hits": [], "new_solutions": 0,
         "first_new_seconds": None, "censored": True},
    ]
    result = summarize(rows)[0]
    assert result["unique_new_pairs"] == 1
    assert result["seconds_per_unique_new_pair"] == 180
    assert result["first_hit_mean_if_uncensored"] is None
    assert result["censored_trials"] == 1


def test_no_hits_does_not_report_a_finite_solution_time():
    result = summarize([{
        "L": 44, "method": "reference", "elapsed": 60, "moves": 600,
        "hits": [], "new_solutions": 0, "first_new_seconds": None, "censored": True,
    }])[0]
    assert result["seconds_per_unique_new_pair"] is None
    assert result["moves_per_second"] == 10


def test_import_uses_verified_deduplicating_production_writer(tmp_path):
    rows = [{"L": 4, "hits": [{"A": "0000", "B": "0011"}]}]
    assert save_verified_discoveries(rows, tmp_path) == {4: 1}
    saved = (tmp_path / "4.txt").read_text()
    assert "nonzero shifts=1,3" in saved
    assert save_verified_discoveries(rows, tmp_path) == {4: 0}
    assert save_verified_discoveries(
        [{"L": 4, "hits": [{"A": "0011", "B": "0000"}]}], tmp_path) == {4: 0}
    assert (tmp_path / "4.txt").read_text() == saved
    with pytest.raises(RuntimeError):
        save_verified_discoveries(
            [{"L": 4, "hits": [{"A": "0000", "B": "0000"}]}], tmp_path)
    assert (tmp_path / "4.txt").read_text() == saved


@pytest.mark.parametrize("already_known", [False, True])
def test_first_new_trial_stops_for_novel_pairs_only(tmp_path, already_known):
    executable = tmp_path / "fake_search"
    executable.write_text(
        "#!" + sys.executable + "\n"
        "import signal, sys\n"
        "def stop(*_args):\n"
        "    print('搜尋統計：restart=1, moves=1, swap evaluations=0, valid results=1', flush=True)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "print('PQCP_CANDIDATE L=4 k=1 sign=1 energy=0 a=0000 b=0011', flush=True)\n"
        "while True: signal.pause()\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    known = {pair_key("0000", "0011")} if already_known else set()
    result = trial(executable, 4, 1, 0.2 if already_known else 3,
                   1, tmp_path / "unused_bank", tmp_path, known, first_new=True)
    assert result["censored"] is already_known
    assert result["new_solutions"] == (0 if already_known else 1)
    if not already_known:
        assert result["elapsed"] < 2
