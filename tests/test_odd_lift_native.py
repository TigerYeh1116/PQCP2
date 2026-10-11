"""Exhaustive literal original binary orientations validate the odd lift transform."""
from itertools import product

import pytest
import torch

from solver.compressed_fkm import lift_factor2_compressed
from solver.correlation import full_correlation_profile
from solver.odd_lift_native import solve_odd_lift_native
from solver.structured_energy import fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.torch_compressed_search import search_compressed


def literal_lifts(a, b, target):
    zero_a, zero_b = a.count(0), b.count(0)
    expected = set()
    for bits in product((0, 1), repeat=zero_a+zero_b):
        words = (lift_factor2_compressed(a, bits[:zero_a]), lift_factor2_compressed(b, bits[zero_a:]))
        if (tuple(full_correlation_profile(*words)) == target_profile(target.L, target.k, target.eta)
                and pair_content(*words) == (target.a_even_ones, target.a_odd_ones,
                                            target.b_even_ones, target.b_odd_ones)):
            expected.add(words)
    return expected


def test_every_small_coarse_pair_against_all_original_binary_lifts():
    words = list(product((-2, 0, 2), repeat=3))
    saw_sat = False
    tested = 0
    for target in canonical_target_content_profiles(6):
        wanted = fold_profile(target_profile(6, target.k, target.eta), 2)
        feasible = []
        for e, o in ((target.a_even_ones, target.a_odd_ones), (target.b_even_ones, target.b_odd_ones)):
            feasible.append([w for w in words if 2*w.count(-2)+w.count(0) == e+o
                             and 0 <= e-w.count(-2) <= w.count(0)])
        for a in feasible[0]:
            for b in feasible[1]:
                if tuple(x+y for x, y in zip(pacf(a), pacf(b))) != wanted:
                    continue
                expected = literal_lifts(a, b, target)
                out = solve_odd_lift_native(a, b, target)
                assert out['status'] == ('LIFT_SAT' if expected else 'LIFT_UNSAT')
                if expected:
                    assert out['verified'] and out['words'] in expected
                    saw_sat = True
                tested += 1
    assert saw_sat and tested > 10


def test_l14_real_coarse_jobs_and_assignment_caps_match_literal_lifts():
    ts = canonical_target_content_profiles(14)
    profiles = tuple(ts[i % len(ts)] for i in range(64))
    result = search_compressed(profiles, device=torch.device('cpu'), seed=791, steps=8)
    assert result.jobs
    saw_cap = False
    for job in result.jobs:
        expected = literal_lifts(job.compressed_a, job.compressed_b, job.target)
        full = solve_odd_lift_native(job.compressed_a, job.compressed_b, job.target)
        assert full['status'] == ('LIFT_SAT' if expected else 'LIFT_UNSAT')
        if expected:
            assert full['words'] in expected
        partial = solve_odd_lift_native(job.compressed_a, job.compressed_b, job.target, max_assignments=1)
        if partial['status'] == 'LIFT_UNSAT':
            assert not expected
        saw_cap |= partial['status'] == 'LIFT_UNKNOWN'
    assert saw_cap


@pytest.mark.parametrize('arguments', [dict(max_assignments=0), dict(max_assignments=True)])
def test_invalid_native_lift_budgets_rejected(arguments):
    ts = canonical_target_content_profiles(6)
    result = search_compressed((ts[0],)*8, device=torch.device('cpu'), seed=17, steps=1)
    job = result.jobs[0]
    with pytest.raises(ValueError):
        solve_odd_lift_native(job.compressed_a, job.compressed_b, job.target, **arguments)
