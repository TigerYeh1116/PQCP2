"""Literal lifts, not solver counts, are the enumeration correctness oracle."""

from dataclasses import asdict, replace
from itertools import product
import json

import pytest

from experiments.benchmark_lift_enumeration import inventory, summarize
from solver import lift_enumeration as enumeration
from solver import staged_completion as staged
from solver.compressed_completion import CompressedLiftJob
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile
from solver.solution_equivalence import pqcp_equivalence_key
from solver.structured_energy import compress_binary_sequence, target_profile
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def job14():
    # Fixed compressed patterns from the fresh L14 experiment, NOT full
    # solution seeds. This job has four raw lifts in two exact full orbits.
    return CompressedLiftJob((2, 0, 0, 2, 2, 0, 0), (0, 2, 0, 2, -2, -2, 0),
        TargetContentProfile(14, 1, 1, 1, 3, 3, 4), (0,)*14, (0,)*14, 2)


def literal_lifts(job):
    """Enumerate EVERY zero orientation, audit with full baseline arithmetic."""
    d = job.target.L // 2
    width = job.compressed_a.count(0) + job.compressed_b.count(0)
    expected = {}
    t = job.target
    for row in product((0, 1), repeat=width):
        values = iter(row)
        pair = []
        for compressed in (job.compressed_a, job.compressed_b):
            word = [0] * (2*d)
            for i, c in enumerate(compressed):
                if c:
                    word[i] = word[i+d] = int(c == -2)
                else:
                    word[i] = next(values)
                    word[i+d] = 1-word[i]
            pair.append(tuple(word))
        if (tuple(full_correlation_profile(*pair)) == target_profile(t.L, t.k, t.eta)
                and pair_content(*pair) == (t.a_even_ones, t.a_odd_ones, t.b_even_ones, t.b_odd_ones)):
            expected[row] = tuple(pair)
    return expected


def test_all_lifts_match_literal_oracle_and_classes():
    job = job14()
    expected = literal_lifts(job)
    result = enumeration.enumerate_lifts(job, max_models=32, seconds=10)
    assert result.terminal == "EXHAUSTED"
    assert len(result.models) == len(expected) == 4
    assert {tuple(m["orientation"]) for m in result.models} == set(expected)
    assert set(result.equivalence_keys) == {pqcp_equivalence_key(*pair) for pair in expected.values()}
    assert len(result.equivalence_keys) == 2  # never count raw SATs as new classes
    for model in result.models:
        assert verify_pqcp(model["A"], model["B"]).is_valid
        assert model["profile"] == full_correlation_profile(model["A"], model["B"])


@pytest.mark.parametrize("length", [4, 6])
def test_small_complete_binary_universe(length):
    jobs = {}
    for bits in product((0, 1), repeat=2*length):
        a, b = bits[:length], bits[length:]
        check = verify_pqcp(a, b)
        if not check.is_valid:
            continue
        k = min(check.nonzero_shifts)
        target = TargetContentProfile(length, k, check.profile[k]//4, *pair_content(a, b))
        job = CompressedLiftJob(compress_binary_sequence(a, 2), compress_binary_sequence(b, 2),
                                target, (0,)*length, (0,)*length, 2)
        jobs.setdefault(job.canonical_key, job)
    assert jobs
    for job in jobs.values():
        result = enumeration.enumerate_lifts(job, max_models=128, seconds=10)
        assert result.terminal == "EXHAUSTED"
        assert set(result.exclusions) == set(literal_lifts(job))


def test_resume_exclusions_and_model_cap():
    job = job14()
    first = enumeration.enumerate_lifts(job, max_models=1, seconds=10)
    assert first.terminal == "MODEL_LIMIT"
    rest = enumeration.enumerate_lifts(job, max_models=32, seconds=10,
                                      excluded_orientations=first.exclusions)
    assert rest.terminal == "EXHAUSTED"
    assert len(rest.models) == 3
    assert set(rest.exclusions) == set(literal_lifts(job))


@pytest.mark.parametrize("gf2", [True, False])
def test_exclusions_block_only_exact_model_with_both_encodings(gf2):
    job = job14()
    expected = literal_lifts(job)
    blocked = list(expected)[:-1]
    result = solve_compressed_lift(job.compressed_a, job.compressed_b, 1, 1, 1000, job.target,
                                   orientation_preprocessing=gf2, excluded_orientations=blocked)
    assert result.status == "SAT" and result.excluded_models == 3
    assert enumeration.orientation_assignment(job, result.a, result.b) == list(expected)[-1]


def test_navigation_cannot_return_an_excluded_model(monkeypatch):
    job = job14()
    expected = literal_lifts(job)
    def fail(*args, **kwargs):
        pytest.fail("navigation should be bypassed when models are excluded")
    monkeypatch.setattr("solver.orientation_search.search_orientations", fail)
    result = solve_compressed_lift(job.compressed_a, job.compressed_b, 1, 1, 1000, job.target,
        orientation_search_steps=128, excluded_orientations=list(expected))
    assert result.status == "UNSAT"


@pytest.mark.parametrize("bad", [[(0,)], [(0,)*6 + (2,)], [(False,)*7]])
def test_invalid_exclusions_rejected(bad):
    job = job14()
    with pytest.raises(ValueError):
        enumeration.enumerate_lifts(job, excluded_orientations=bad)
    with pytest.raises(ValueError):
        solve_compressed_lift(job.compressed_a, job.compressed_b, 1, 1, 1000, job.target,
                              excluded_orientations=bad)


def test_unknown_and_budget_are_not_unsat(monkeypatch):
    job = job14()
    real = solve_compressed_lift(job.compressed_a, job.compressed_b, 1, 1, 1000, job.target)
    unknown = replace(real, status="UNKNOWN", a=None, b=None, profile=None, verified=False)
    monkeypatch.setattr(enumeration, "solve_compressed_lift", lambda *a, **k: unknown)
    assert enumeration.enumerate_lifts(job).terminal == "UNKNOWN"
    ticks = iter((0., 2., 2.))
    monkeypatch.setattr(enumeration, "perf_counter", lambda: next(ticks))
    result = enumeration.enumerate_lifts(job, seconds=1)
    assert result.terminal == "TIME_LIMIT" and result.checks == 0


def test_callback_failure_is_not_swallowed():
    def fail(model):
        assert model["A"] and model["B"]
        raise OSError("disk failure")
    with pytest.raises(OSError, match="disk failure"):
        enumeration.enumerate_lifts(job14(), on_model=fail)


def make_store(tmp_path, **options):
    store = staged.CompletionWorkStore(tmp_path, 14, 123,
        staged.SlicePolicy(lift_max_models=options.pop("lift_max_models", 8),
                           lift_extra_ms=options.pop("lift_extra_ms", 10000), **options))
    key = store.register("lift", asdict(job14()))
    return store, key


def test_staged_resume_preserves_every_model_and_official_dedup(tmp_path):
    store, key = make_store(tmp_path)
    first = store.advance_lift(key)
    assert first["solved"] and first["phase"] == "READY"
    assert len(store.read(key)["lift_enumeration"]["models"]) == 1
    store = staged.CompletionWorkStore(tmp_path, 14, 123, store.policy)
    assert store.recover() == [key]
    reports = [first]
    for _ in range(10):
        result = store.advance_lift(key)
        reports.append(result)
        if result["phase"] != "READY":
            break
    record = store.read(key)
    assert record["phase"] == "SAT"  # residual UNSAT never erases original SAT!
    assert record["lift_enumeration"]["status"] == "EXHAUSTED"
    assert len(record["lift_enumeration"]["models"]) == 4
    assert sum(r["new_solution_written"] for r in reports) == 2
    assert all(r["z3_calls"] <= 1 for r in reports)
    keys, _ = inventory(tmp_path / "14.txt", 14)
    assert len(keys) == 2
    assert store.recover() == []
    last_snapshot = json.loads(open(reports[-2]["snapshot"]).read())
    assert last_snapshot["status"] == "SAT"


def test_staged_bounded_mode_stops_without_claiming_exhaustion(tmp_path):
    store, key = make_store(tmp_path, lift_max_models=2)
    assert store.advance_lift(key)["phase"] == "READY"
    assert store.advance_lift(key)["phase"] == "SAT"
    progress = store.read(key)["lift_enumeration"]
    assert progress["status"] == "MODEL_LIMIT" and len(progress["models"]) == 2


def test_optional_extra_timeout_parks_preserving_first_solution(tmp_path, monkeypatch):
    store, key = make_store(tmp_path)
    store.advance_lift(key)
    job = job14()
    real = solve_compressed_lift(job.compressed_a, job.compressed_b, 1, 1, 1000, job.target)
    unknown = replace(real, status="UNKNOWN", a=None, b=None, profile=None, verified=False)
    monkeypatch.setattr(staged, "solve_compressed_lift", lambda *a, **k: unknown)
    result = store.advance_lift(key)
    assert result["phase"] == "WAITING" and result["status"] == "LIFT_ENUM_UNKNOWN"
    assert len(store.read(key)["lift_enumeration"]["models"]) == 1
    assert store.recover() == [key]
    monkeypatch.undo()
    assert store.advance_lift(key)["solved"]


def test_failed_persistence_keeps_previous_cursor(tmp_path, monkeypatch):
    store, key = make_store(tmp_path)
    store.advance_lift(key)
    before = store.path(key).read_bytes()
    def fail(*a, **k):
        raise OSError("disk failure")
    monkeypatch.setattr(staged, "persist_compressed_lift_result", fail)
    with pytest.raises(OSError, match="disk failure"):
        store.advance_lift(key)
    assert store.path(key).read_bytes() == before


def test_staged_default_remains_first_solution_only(tmp_path):
    store, key = make_store(tmp_path, lift_max_models=1)
    assert store.advance_lift(key)["phase"] == "SAT"
    assert "lift_enumeration" not in store.read(key)


def test_extra_time_limit_keeps_sat_and_stops_queue(tmp_path, monkeypatch):
    store, key = make_store(tmp_path, lift_extra_ms=200)
    store.advance_lift(key)
    ticks = iter((0., 1., 1.))
    monkeypatch.setattr(staged, "perf_counter", lambda: next(ticks))
    result = store.advance_lift(key)
    assert result["solved"] and result["phase"] == "SAT"
    assert result["lift_enumeration_status"] == "TIME_LIMIT"
    assert len(store.read(key)["lift_enumeration"]["models"]) == 2


def test_disabling_feature_on_resume_preserves_sat_without_extra_checks(tmp_path):
    store, key = make_store(tmp_path)
    store.advance_lift(key)
    disabled = staged.CompletionWorkStore(tmp_path, 14, 123, staged.SlicePolicy())
    assert disabled.recover() == []
    assert disabled.read(key)["phase"] == "SAT"
    assert len(disabled.read(key)["lift_enumeration"]["models"]) == 1


def test_report_union_counts_classes_not_raw_models():
    def row(keys):
        return dict(models=[dict(equivalence_key=[k], elapsed=.1) for k in keys],
                    equivalence_keys=list(set(keys)), elapsed=.2, terminal="EXHAUSTED")
    result = summarize([row([1, 1, 2]), row([2, 3])], {(1,)})
    assert result["raw_models"] == 5 and result["all_model_global_classes"] == 3
    assert result["first_model_global_classes"] == 2
    assert result["extra_new_to_inventory"] == 1
