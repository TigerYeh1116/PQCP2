from threading import Event

import torch

from solver.compressed_repair_pipeline import repair_search_result
from solver.structured_energy import fold_profile, target_profile, integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.torch_compressed_search import search_compressed


def fixture():
    targets = canonical_target_content_profiles(14)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    result = search_compressed(profiles, device=torch.device('cpu'), seed=789, steps=1)
    return profiles, result


def test_repair_exports_exact_jobs_and_deduplicates():
    profiles, result = fixture()
    original = len(result.jobs)
    records = repair_search_result(result, profiles, top=32)
    assert len(result.jobs) > original  # Real nonzero -> exact compressed transition.
    assert sum(r['new_job'] for r in records) == len(result.jobs) - original
    for job in result.jobs:
        assert tuple(map(sum, zip(pacf(job.compressed_a), pacf(job.compressed_b)))) == fold_profile(
            target_profile(14, job.target.k, job.target.eta), 2)
        t = job.target
        assert pair_content(job.source_a, job.source_b) == (
            t.a_even_ones, t.a_odd_ones, t.b_even_ones, t.b_odd_ones)
    count = len(result.jobs)
    repeated = repair_search_result(result, profiles, top=32)
    assert len(result.jobs) == count
    assert not any(r['new_job'] for r in repeated)


def test_disabled_and_cancelled_do_not_invoke_native(monkeypatch):
    profiles, result = fixture()
    import solver.compressed_repair_pipeline as pipeline
    monkeypatch.setattr(pipeline, 'repair_two_swaps', lambda *a: (_ for _ in ()).throw(AssertionError()))
    assert repair_search_result(result, profiles, top=0) == []
    event = Event()
    event.set()
    assert repair_search_result(result, profiles, top=8, stop_event=event) == []


def test_top_budget_and_cancellation_between_candidates(monkeypatch):
    profiles, result = fixture()
    assert len(repair_search_result(result, profiles, top=2)) == 2
    import solver.compressed_repair_pipeline as pipeline
    native = pipeline.repair_two_swaps
    event = Event()
    def stop_after_one(*args):
        value = native(*args)
        event.set()
        return value
    monkeypatch.setattr(pipeline, 'repair_two_swaps', stop_after_one)
    assert len(repair_search_result(result, profiles, top=8, stop_event=event)) == 1


def test_advanced_repair_handoff_preserves_multiple_jobs_and_cap_status():
    from solver.compressed_repair_pipeline import advanced_repair_search_result
    profiles, result = fixture()
    original = len(result.jobs)
    records = advanced_repair_search_result(result, profiles, transfer_top=2, mitm_top=2, max_results=2)
    assert {r['method'] for r in records} == {'content_transfer', 'mitm_two_per_side'}
    assert len(result.jobs) > original
    assert sum(r['new_jobs'] for r in records) == len(result.jobs)-original
    for r in records:
        if r['method'] == 'mitm_two_per_side':
            assert len(r['result']['words']) <= 2
            assert r['result']['complete'] == (r['result']['status'] != 'SAT_TRUNCATED')
    event=Event();event.set()
    assert advanced_repair_search_result(result,profiles,transfer_top=8,mitm_top=1,stop_event=event)==[]


def test_native_continuation_exports_hits_and_never_worsens_best():
    from solver.compressed_repair_pipeline import refine_search_result_native
    profiles,result=fixture()
    old=result.energies.clone();count=len(result.jobs)
    records=refine_search_result_native(result,profiles,top=32,moves=256,seed=123)
    assert torch.all(result.energies<=old)
    assert len(result.jobs)>count
    assert sum(r['new_job'] for r in records)==len(result.jobs)-count
    for job in result.jobs:
        assert tuple(map(sum,zip(pacf(job.compressed_a),pacf(job.compressed_b))))==fold_profile(
            target_profile(14,job.target.k,job.target.eta),2)
    assert refine_search_result_native(result,profiles,top=0)==[]
    event=Event();event.set()
    assert refine_search_result_native(result,profiles,top=8,stop_event=event)==[]


def test_support_repair_handoff_and_misses_preserve_reference():
    from solver.compressed_repair_pipeline import repair_search_result_support
    profiles, result = fixture()
    before = result.words.clone(); energy = result.energies.clone(); count = len(result.jobs)
    records = repair_search_result_support(result, profiles, top=32, max_supports=1000)
    assert len(result.jobs) > count
    assert sum(r['new_job'] for r in records) == len(result.jobs)-count
    for row in records:
        lane = row['lane']
        if row['result']['words'] is None:
            assert torch.equal(result.words[lane], before[lane])
            assert result.energies[lane] == energy[lane]
        else:
            assert result.energies[lane] == 0
    for job in result.jobs:
        assert tuple(map(sum, zip(pacf(job.compressed_a), pacf(job.compressed_b)))) == fold_profile(
            target_profile(14, job.target.k, job.target.eta), 2)
        t = job.target
        assert pair_content(job.source_a, job.source_b) == (
            t.a_even_ones, t.a_odd_ones, t.b_even_ones, t.b_odd_ones)
    count = len(result.jobs)
    repeated = repair_search_result_support(result, profiles, top=32, max_supports=1000)
    assert len(result.jobs) == count and not any(r['new_job'] for r in repeated)
    assert repair_search_result_support(result, profiles, top=0) == []
    event = Event(); event.set()
    assert repair_search_result_support(result, profiles, stop_event=event) == []
