"""Finite proof coverage and conservative sampling/solver integration."""
from solver.l86_content_certificate import l86_content_certificate
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents, search_paired_contents
from solver.ternary_support_search import complete_support_signs


def test_runtime_finite_certificate_covers_all_four_sparse_cases():
    report = l86_content_certificate()
    assert report.verified and report.minimum_weight >= 7
    assert report.cases == ((5, -1, (0, 17), 2), (5, 1, (16, 1), 2),
                            (6, 1, (0, 17), 3438), (6, -1, (16, 1), 3438))
    assert report.assignments > 0


def test_search_sampling_keeps_raw_count_feasibility_and_removes_only_certified_sizes():
    for target in canonical_target_content_profiles(86):
        raw = paired_contents(target)
        filtered = search_paired_contents(target, binary_content_pruning=False)
        minimum = l86_content_certificate().minimum_weight
        assert len(filtered) == (14 if minimum == 8 else 15)
        assert filtered == tuple(pair for pair in raw if all(n+p >= minimum for n, _, p in pair))
        assert search_paired_contents(target, content_pruning=False) == raw
    for target in canonical_target_content_profiles(14):
        assert search_paired_contents(target) == paired_contents(target)


def test_exact_support_completion_rejects_certified_content_before_enumeration():
    target = next(t for t in canonical_target_content_profiles(86) if t.k == 2 and t.eta == -1)
    a, b = tuple([0]*38+[1]*5), tuple([-1]*17+[0]*5+[1]*21)
    out = complete_support_signs(a, b, target, (1116169, ((1 << 43)-1)^537395331))
    assert out['status'] == 'UNSAT_FIXED_SUPPORT'
    assert out['assignments'] == 0 and out['minimum_support'] == l86_content_certificate().minimum_weight
    legacy = complete_support_signs(a, b, target, (1116169, ((1 << 43)-1)^537395331),
                                    content_pruning=False)
    assert legacy['status'] == out['status']
    assert legacy['assignments'] > 0


def test_incomplete_certificate_falls_back_to_analytic_bound(monkeypatch):
    import solver.l86_content_certificate as module
    module.l86_content_certificate.cache_clear()
    try:
        monkeypatch.setattr(module, 'solve_sign_system', lambda *a, **k:
                            dict(status='UNKNOWN', assignments=1, words=None))
        report = module.l86_content_certificate()
        assert not report.verified and report.minimum_weight == 5
        for target in canonical_target_content_profiles(86):
            assert search_paired_contents(target) == tuple(pair for pair in paired_contents(target)
                if all(n+p >= 5 for n, _, p in pair))
    finally:
        module.l86_content_certificate.cache_clear()


def test_recorded_weight7_tier_authenticates_scope_and_independently_checks_the_refined_leaf(tmp_path):
    import json
    from pathlib import Path
    from solver.l86_content_certificate import recorded_weight7_cases
    data = json.loads((Path(__file__).parent/'fixtures/l86_coarse_weight7_runs.json').read_text())
    for name, raw in data.items():
        path = tmp_path/'results'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
    assert recorded_weight7_cases(tmp_path) == ((7, -1, (1, 16), 130142), (7, 1, (15, 2), 130142))
    name = next(iter(data))
    (tmp_path/'results'/name).write_text(data[name]+' ')
    assert not recorded_weight7_cases(tmp_path)
