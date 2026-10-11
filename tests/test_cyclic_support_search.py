"""Full signed-word oracle for the new support scheduling and lift handoff."""
from itertools import permutations
from threading import Event

import pytest
import torch

from solver.compressed_repair import _validated_reference
from solver.compressed_repair_pipeline import repair_search_result_support
from solver.cyclic_support_search import repair_cyclic_supports
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf, fold_profile, target_profile
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.ternary_support_search import support_mask
from solver.torch_compressed_search import paired_contents, search_compressed


def test_cyclic_sign_solver_matches_all_literal_nearest_solutions():
    saw_sat = False
    for target in canonical_target_content_profiles(6):
        for contents in paired_contents(target):
            words = [sorted(set(permutations([-1]*n+[0]*z+[1]*p))) for n, z, p in contents]
            a, b = words[0][0], words[1][0]
            _, _, d, wanted = _validated_reference(a, b, target)
            result = repair_cyclic_supports(a, b, target, max_supports=1000)
            reference_masks = tuple(support_mask(w) for w in (a, b))
            possible = set()
            for aa in words[0]:
                for bb in words[1]:
                    if tuple(x+y for x, y in zip(pacf(aa), pacf(bb))) != wanted:
                        continue
                    masks = tuple(support_mask(w) for w in (aa, bb))
                    for g in result['generation']:
                        fixed = g['fixed_side']
                        if (masks[fixed] == reference_masks[fixed]
                                and bin(masks[1-fixed] ^ reference_masks[1-fixed]).count('1') == g['distance']):
                            possible.add((aa, bb))
            if possible:
                assert result['status'] == 'SAT_COMPRESSED'
                assert result['words'] in possible
                saw_sat = True
            else:
                assert result['status'] == 'BOUNDED_MISS'
    assert saw_sat


def test_cyclic_handoff_exports_audited_jobs_and_keeps_misses():
    targets = canonical_target_content_profiles(22)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    result = search_compressed(profiles, device=torch.device('cpu'), seed=1, steps=1)
    before = result.words.clone()
    count = len(result.jobs)
    records = repair_search_result_support(result, profiles, method='cyclic_norm', top=32, max_supports=1000)
    assert len(result.jobs) > count
    assert sum(r['new_job'] for r in records) == len(result.jobs)-count
    for r in records:
        if r['result']['words'] is None:
            assert torch.equal(result.words[r['lane']], before[r['lane']])
    for job in result.jobs:
        assert tuple(x+y for x, y in zip(pacf(job.compressed_a), pacf(job.compressed_b))) == fold_profile(
            target_profile(22, job.target.k, job.target.eta), 2)
        t = job.target
        assert pair_content(job.source_a, job.source_b) == (
            t.a_even_ones, t.a_odd_ones, t.b_even_ones, t.b_odd_ones)


def test_cancelled_and_unknown_signs_do_not_claim_unsat(monkeypatch):
    import solver.cyclic_support_search as module
    target = canonical_target_content_profiles(6)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in paired_contents(target)[0])
    event = Event()
    event.set()
    monkeypatch.setattr(module, 'parity_support_family', lambda *args: pytest.fail('generated after cancellation'))
    assert repair_cyclic_supports(a, b, target, stop_event=event)['status'] == 'UNKNOWN'
    monkeypatch.undo()
    monkeypatch.setattr(module, 'complete_support_signs', lambda *args, **kwargs: dict(status='UNKNOWN', words=None))
    assert repair_cyclic_supports(a, b, target)['status'] == 'UNKNOWN'


def test_l86_default_does_not_stop_at_a_proved_unliftable_coarse_hit():
    from tests.test_compressed_lift_native import A, B, TARGET
    a, b = tuple(v//2 for v in A), tuple(v//2 for v in B)
    legacy = repair_cyclic_supports(a, b, TARGET, full_completion=False)
    assert legacy['status'] == 'SAT_COMPRESSED'
    full = repair_cyclic_supports(a, b, TARGET)
    assert full['completion_scope'] == 'binary' and full['status'] == 'BOUNDED_MISS'
    assert full['words'] is None and full['full_words'] is None
    assert len(full['attempts']) > 1
    assert any(g['distance'] and g['distance'] > 0 for g in full['generation'])
    assert any(g['distance'] == 0 for g in full['exhausted_layers'])
    assert any(r['result']['excluded_by'] == 'difference' and
               r['result']['attempts']['coarse']['status'] == 'SAT_COMPRESSED' for r in full['attempts'])


def test_full_binary_handoff_preserves_verified_sources_even_for_duplicate_compressions(monkeypatch):
    from solver.compressed_completion import CompressedLiftJob
    from solver.torch_compressed_search import CompressedSearchResult, representative_lift, ternary_profile
    from solver.verifier import verify_pqcp
    targets = canonical_target_content_profiles(22)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    result = search_compressed(profiles, device=torch.device('cpu'), seed=1, steps=1)
    records = repair_search_result_support(result, profiles, method='cyclic_norm', top=32,
                                          max_supports=1000, full_completion=True)
    hits = [r for r in records if r['result']['full_words'] is not None]
    assert hits
    for job in result.jobs:
        # Initial coarse-only jobs may remain; inspect native full sources.
        if any(job.source_a == h['result']['full_words'][0] and job.source_b == h['result']['full_words'][1] for h in hits):
            assert verify_pqcp(job.source_a, job.source_b).is_valid
    hit = hits[0]
    target = profiles[hit['lane']]
    a, b = hit['result']['words']
    ca, cb = tuple(2*v for v in a), tuple(2*v for v in b)
    old_job = CompressedLiftJob(ca, cb, target,
        representative_lift(ca, target.a_even_ones, target.a_odd_ones),
        representative_lift(cb, target.b_even_ones, target.b_odd_ones))
    tensor = torch.tensor([hit['reference']], dtype=torch.float32)
    duplicate = CompressedSearchResult(tensor, ternary_profile(tensor), torch.tensor([1.]), [old_job], 0, 0, 0)
    monkeypatch.setattr('solver.cyclic_support_search.repair_cyclic_supports', lambda *_a, **_k: hit['result'])
    changed = repair_search_result_support(duplicate, (target,), method='cyclic_norm', full_completion=True)
    assert len(duplicate.jobs) == 1 and not changed[0]['new_job']
    assert changed[0]['full_source_upgraded']
    assert (duplicate.jobs[0].source_a, duplicate.jobs[0].source_b) == hit['result']['full_words']
    assert verify_pqcp(duplicate.jobs[0].source_a, duplicate.jobs[0].source_b).is_valid


@pytest.mark.parametrize('obstruction', ['unknown', 'truncated'])
def test_unresolved_or_truncated_layers_never_advance(monkeypatch, obstruction):
    import solver.cyclic_support_search as module
    target = canonical_target_content_profiles(6)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in paired_contents(target)[0])
    native_nearest = module.nearest_supports
    floors = []
    def nearest(*args, **kwargs):
        floors.append(kwargs.get('minimum_distance', 0))
        result = native_nearest(*args, **kwargs)
        if obstruction == 'truncated':
            result['complete_nearest'] = False
        return result
    monkeypatch.setattr(module, 'nearest_supports', nearest)
    monkeypatch.setattr('solver.pqcp_support_solver.complete_pqcp_supports', lambda *_a, **_k: dict(
        status='UNKNOWN' if obstruction == 'unknown' else 'UNSAT_BINARY_SUPPORT', words=None,
        attempts={}, verified=False, excluded_by=None))
    result = repair_cyclic_supports(a, b, target, full_completion=True)
    assert floors == [0, 0] and not result['exhausted_layers']
    assert result['status'] == ('UNKNOWN' if obstruction == 'unknown' else 'BOUNDED_MISS')


def test_exact_but_unliftable_lane_is_not_skipped_by_full_support_handoff():
    from tests.test_compressed_lift_native import A, B, TARGET
    from solver.torch_compressed_search import CompressedSearchResult, ternary_profile
    tensor = torch.tensor([[tuple(v//2 for v in A), tuple(v//2 for v in B)]], dtype=torch.float32)
    result = CompressedSearchResult(tensor, ternary_profile(tensor), torch.tensor([0.]), [], 0, 0, 0)
    records = repair_search_result_support(result, (TARGET,), method='cyclic_norm')
    assert len(records) == 1 and records[0]['original_energy'] == 0
    assert len(records[0]['result']['attempts']) > 1


@pytest.mark.parametrize('kwargs', [dict(max_supports=0), dict(max_supports=True),
    dict(max_supports=4097), dict(max_assignments=0), dict(max_assignments=True)])
def test_invalid_budgets_rejected(kwargs):
    target = canonical_target_content_profiles(6)[0]
    a, b = (tuple([-1]*n+[0]*z+[1]*p) for n, z, p in paired_contents(target)[0])
    with pytest.raises(ValueError):
        repair_cyclic_supports(a, b, target, **kwargs)
