"""Free-space proposals remain necessary-condition solutions, never UNSAT proofs."""

from dataclasses import asdict
from itertools import product

import pytest
import torch

from solver.compressed_fkm import lift_factor2_compressed
from solver.compressed_lift import solve_compressed_lift, CompressedLiftResult
from solver.compressed_completion import run_compressed_lift_job
from solver.correlation import full_correlation_profile
from solver.orientation_gf2 import odd_orientation_system
from solver.orientation_search import search_orientations
from solver.staged_completion import CompletionWorkStore, SlicePolicy
from solver.structured_energy import target_profile, compress_binary_sequence
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.torch_compressed_search import search_compressed
from solver.verifier import verify_pqcp


@pytest.fixture(scope="module")
def jobs():
    ts = canonical_target_content_profiles(14)
    result = search_compressed(tuple(ts[i % len(ts)] for i in range(16)),
                              device=torch.device("cpu"), seed=123, steps=32, candidates=4)
    assert result.jobs
    return result.jobs


def test_proposals_satisfy_gf2_and_only_content_feasible_enter_search(jobs):
    for job in jobs:
        trace = []
        result = search_orientations(job.compressed_a, job.compressed_b, job.target, steps=32, seed=42, trace=trace)
        t = target_profile(14, job.target.k, job.target.eta)
        system = odd_orientation_system(job.compressed_a, job.compressed_b, tuple((t[u]-t[u+7])//4 for u in range(7)))
        assert result.status in ("SAT", "MISS")
        for _, pair, feasible, energy in trace:
            assert tuple(compress_binary_sequence(w, 2) for w in pair) == (job.compressed_a, job.compressed_b)
            assignment = sum(pair[s][i] << j for j, (s, i) in enumerate(system.variables))
            assert all(bin(mask & assignment).count("1") % 2 == rhs for _, mask, rhs, _ in system.equations)
            assert feasible == (pair_content(*pair) == pair_content(job.source_a, job.source_b))
            if feasible:
                observed = full_correlation_profile(*pair)
                assert energy == sum(((observed[u]-t[u])//2)**2 for u in range(1, 4))
        if result.pair is not None:
            assert pair_content(*result.pair) == pair_content(job.source_a, job.source_b)
        repeated = search_orientations(job.compressed_a, job.compressed_b, job.target, steps=32, seed=42)
        assert (result.status, result.pair, result.evaluations) == (repeated.status, repeated.pair, repeated.evaluations)


def test_z3_with_navigation_agrees_with_literal_lift_oracle(jobs):
    for job in jobs[:6]:
        wanted = target_profile(14, job.target.k, job.target.eta)
        def lifts(c):
            return [lift_factor2_compressed(c, bits) for bits in product((0, 1), repeat=c.count(0))]
        feasible = any(tuple(full_correlation_profile(a, b)) == wanted
                       and pair_content(a, b) == pair_content(job.source_a, job.source_b)
                       for a, b in product(lifts(job.compressed_a), lifts(job.compressed_b)))
        results = [solve_compressed_lift(job.compressed_a, job.compressed_b, job.target.k, job.target.eta,
                                        2000, job.target, orientation_search_steps=steps) for steps in (0, 1, 32)]
        assert all(r.status == ("SAT" if feasible else "UNSAT") for r in results)
        for r in results:
            if r.verified:
                assert verify_pqcp(r.a, r.b).is_valid


def test_bounded_miss_still_calls_z3(jobs, monkeypatch):
    from solver.orientation_search import OrientationSearchResult
    monkeypatch.setattr("solver.orientation_search.search_orientations",
                        lambda *a, **kw: OrientationSearchResult("MISS", None, None, 1, 0, .001))
    job = jobs[0]
    result = solve_compressed_lift(job.compressed_a, job.compressed_b, job.target.k, job.target.eta,
                                  2000, job.target, orientation_search_steps=1)
    assert result.z3_executed and result.status == "SAT"
    assert result.orientation_search_evaluations == 1


def test_staged_resume_does_not_repeat_navigation(jobs, tmp_path, monkeypatch):
    job = jobs[0]
    seen = []
    def unknown(*args, **kwargs):
        seen.append(kwargs)
        return CompressedLiftResult("UNKNOWN", 14, job.target.k, job.target.eta, 7, 3, 4,
                                    0., 0., True, orientation_search_evaluations=1,
                                    orientation_hint=(job.source_a, job.source_b))
    monkeypatch.setattr("solver.staged_completion.solve_compressed_lift", unknown)
    policy = SlicePolicy(initial_ms=1, maximum_ms=4, orientation_search_steps=32)
    store = CompletionWorkStore(tmp_path, 14, 123, policy)
    key = store.register("lift", asdict(job))
    first = store.advance_lift(key)
    assert first["orientation_search_evaluations"] == 1  # retained even on UNKNOWN
    restored = CompletionWorkStore(tmp_path, 14, 123, policy)
    assert key in restored.recover()
    restored.advance_lift(key)
    assert seen[0]["orientation_search_steps"] == 32
    assert seen[1].get("orientation_search_steps", 0) == 0
    assert tuple(map(tuple, seen[1]["preferred_pair"])) == (job.source_a, job.source_b)


def test_verified_navigation_uses_existing_persistence(jobs, tmp_path):
    reports = [run_compressed_lift_job(job, 2000, tmp_path, 123, orientation_search_steps=128)
               for job in jobs[:3]]
    assert any(r["solved"] and not r["z3_executed"] for r in reports)
    assert (tmp_path / "14.txt").is_file()
    assert all(r["solved"] for r in reports)
