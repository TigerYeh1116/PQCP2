"""Lightweight aggregation/observation checks; never run search in pytest."""
import json
import io
import subprocess
import sys

import pytest

from experiments.benchmark_middle_mitm_L44_search import Observation, ObservedOutput, summarize
from tests.test_middle_mitm_L44_record import fixture


def test_clock_has_same_epoch_in_fresh_process():
    from experiments.benchmark_middle_mitm_L44_search import perf_counter
    before = perf_counter()
    child = float(subprocess.check_output([sys.executable, '-c',
        'from time import clock_gettime,CLOCK_MONOTONIC; print(clock_gettime(CLOCK_MONOTONIC))']))
    after = perf_counter()
    assert before <= child <= after


def test_reject_negative_discovery_times():
    with pytest.raises(ValueError, match='timestamp'):
        summarize([dict(first_new_seconds=-1, budget_seconds=60)])


def test_gpu_cycles_are_already_one_based(tmp_path):
    observation = Observation(tmp_path, 0, 60)
    output = ObservedOutput(io.StringIO(), observation)
    try:
        output.write('[MPS] cycle=1 strategy=test\n')
        output.write('[MPS] cycle=2 strategy=test\n')
        assert observation.gpu_cycles == 2
    finally:
        observation.stream.close()


def test_censored_runs_are_not_success_times():
    rows = [dict(method='mitm', budget_seconds=60, new_count=2, first_new_seconds=5,
                 work_count_at_shutdown=8, work_unit='coarse jobs', shutdown_seconds=1),
            dict(method='mitm', budget_seconds=60, new_count=0, first_new_seconds=None,
                 work_count_at_shutdown=4, work_unit='coarse jobs', shutdown_seconds=0)]
    item = summarize(rows)['mitm']
    assert item['censored_runs'] == 1
    assert item['mean_first_new_reached_only'] == 5
    assert item['seconds_per_new'] == 60
    assert item['work_per_new_including_shutdown'] == 6


def test_no_discovery_is_not_zero_seconds():
    row = dict(method='production_mps', budget_seconds=60, new_count=0,
               first_new_seconds=None, work_count_at_shutdown=100,
               work_unit='SA restarts', shutdown_seconds=2)
    item = summarize([row])['production_mps']
    assert item['seconds_per_new'] is None
    assert item['mean_first_new_reached_only'] is None
    assert item['work_per_new_including_shutdown'] is None


def test_observation_keeps_writer_dedup_and_exact_profile(tmp_path, monkeypatch):
    import solver.search_runner as runner
    import experiments.benchmark_middle_mitm_L44_search as bench
    # Restore all patched aliases after testing, as the worker normally exits.
    original = runner.append_verified_solution_if_new
    monkeypatch.setattr(runner, 'append_verified_solution_if_new', original)
    observation = Observation(tmp_path, 100, 60)
    monkeypatch.setattr(bench, 'perf_counter', lambda: 107)
    observation.install()
    result, _ = fixture()
    try:
        assert runner.append_verified_solution_if_new(44, result['A'], result['B'], tmp_path)
        assert not runner.append_verified_solution_if_new(44, result['B'], result['A'], tmp_path)
        event = json.loads((tmp_path/'discoveries.jsonl').read_text())
        assert event['elapsed'] == 7 and event['within_budget']
        assert event['profile'][0] == 88 and event['profile'][2] == -4
    finally:
        import sys
        for name, module in list(sys.modules.items()):
            if name == 'main' or name.startswith('solver.'):
                observed = getattr(module, 'append_verified_solution_if_new', None)
                if getattr(observed, '__name__', '') == 'observed':
                    setattr(module, 'append_verified_solution_if_new', original)
        observation.stream.close()
