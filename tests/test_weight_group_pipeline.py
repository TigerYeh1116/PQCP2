"""Whole sampling/Boolean aggregation controls with real small PQCP hits."""
from experiments import benchmark_weight_group_pipeline as benchmark
from solver.projection_group_targets import group_projection_targets
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.weight_group_targets import group_weight_targets
from solver.verifier import verify_pqcp
import pytest


@pytest.mark.parametrize('paired_c', (False, True))
def test_matched_proposals_decisions_and_full_positive_controls(paired_c):
    choices = tuple((t, c) for t in canonical_target_content_profiles(6)
                    for c in paired_contents(t))
    groups = group_weight_targets(group_projection_targets(choices))
    benchmark.initialize_worker(groups, paired_c)
    positive = False
    for budget in (1, 2**23):
        for number in range(128):
            old = benchmark.trial((number, 123, budget, 'rollback'))
            new = benchmark.trial((number, 123, budget, 'candidate'))
            assert all(old[k] == new[k] for k in
                       ('trial', 'group', 'u', 'sampling', 'status', 'verified'))
            if new['verified']:
                positive = True
                assert verify_pqcp(*old['words']).is_valid
                assert verify_pqcp(*new['words']).is_valid
    assert positive
