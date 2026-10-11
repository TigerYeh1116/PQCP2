"""Lossless bounded histories, exact disk indexes and corruption recovery."""
from types import SimpleNamespace
import tracemalloc

import pytest

from main import _CCompletionBridge
from solver.disk_index import DiskIndex
from solver.search_metrics import SearchMetrics


def test_completion_counts_survive_history_eviction(tmp_path):
    bridge = _CCompletionBridge(SimpleNamespace(repair=False, z3=False), 86, tmp_path)
    for i in range(10000):
        bridge._record_result(dict(status='UNKNOWN', z3_calls=i % 3, solved=i % 11 == 0))
    assert len(bridge._results) == 128
    assert bridge.trigger_count == 10000
    assert bridge.z3_execution_count == sum(i % 3 for i in range(10000))
    assert bridge.valid_result_count == sum(i % 11 == 0 for i in range(10000))
    bridge.close()


def test_exact_index_after_spilling_far_beyond_the_page_cache():
    index = DiskIndex()
    tracemalloc.start()
    try:
        for i in range(20000):
            key = ('lift', (86, i, tuple((i >> j) & 1 for j in range(43))))
            assert index.add(key)
        current, _ = tracemalloc.get_traced_memory()
        # Would require many MB if the full tuple histories were kept in Python.
        assert current < 512*1024
        assert len(index) == 20000
        for i in range(20000):
            assert not index.add(('lift', (86, i, tuple((i >> j) & 1 for j in range(43)))))
        assert len(index) == 20000
        index[('representative', 1)] = dict(key='old', rank=[False, True, 0, 1])
        index[('representative', 1)] = dict(key='new', rank=[True, False, 0, 2])
        assert index.get(('representative', 1))['key'] == 'new'
        assert index.get('missing') is None
    finally:
        tracemalloc.stop(); index.close()


def test_metric_unique_count_remains_exact_after_history_spill(tmp_path):
    metrics = SearchMetrics(tmp_path, 86, 123)
    for i in range(10000):
        metrics.matched(('pair', i))
    for i in range(10000):
        metrics.matched(('pair', i))
    assert metrics.snapshot()['unique_compressed_matches'] == 10000
    metrics.close()
    assert metrics.snapshot()['unique_compressed_matches'] == 10000


def test_metrics_index_error_does_not_stop_search(tmp_path, monkeypatch):
    import sqlite3
    metrics = SearchMetrics(tmp_path, 86, 123)
    def fail(key): raise sqlite3.OperationalError('disk full')
    monkeypatch.setattr(metrics._matched, 'add', fail)
    with pytest.warns(UserWarning, match='index failure'):
        metrics.matched('x')
    assert not metrics.enabled
    metrics.close()


def test_mps_pool_drains_on_exception_and_cpu_does_not_load_foundation(monkeypatch):
    import solver.mps_memory as module
    events = []
    fake = SimpleNamespace(pqcp_autorelease_create=lambda: events.append('create') or 42,
                           pqcp_autorelease_drain=lambda handle: events.append(('drain', handle)))
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module, '_library', lambda: fake)
    with module.autorelease_pool('cpu'):
        pass
    assert events == []
    with pytest.raises(ValueError):
        with module.autorelease_pool('mps'):
            raise ValueError('test')
    assert events == ['create', ('drain', 42)]
    events.clear()
    with module.autorelease_pool('mps'):
        with module.autorelease_pool('mps'):
            pass
    assert events == ['create', ('drain', 42)]


def test_mps_unit_stride_origin_preserves_value_and_analytic_gradient():
    import torch
    matrix = torch.randn(7, 86, dtype=torch.float64, requires_grad=True)
    old = matrix[:, 0].square()/16
    new = matrix[:, 0:1].square().squeeze(-1)/16
    assert torch.equal(old, new)
    old_grad = torch.autograd.grad(old.sum(), matrix, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new.sum(), matrix)[0]
    assert torch.equal(old_grad, new_grad)


def test_shared_pool_scope_is_thread_local(monkeypatch):
    import threading
    import solver.mps_memory as module
    events = []
    fake = SimpleNamespace(pqcp_autorelease_create=lambda: events.append(('create', threading.get_ident())) or 42,
                           pqcp_autorelease_drain=lambda handle: events.append(('drain', threading.get_ident())))
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module, '_library', lambda: fake)
    def other_thread():
        with module.autorelease_pool('mps'):
            pass
    with module.autorelease_pool('mps'):
        thread = threading.Thread(target=other_thread)
        thread.start(); thread.join()
    assert sum(kind == 'create' for kind, _ in events) == 2
    assert sum(kind == 'drain' for kind, _ in events) == 2
    for _, identity in events:
        assert sum(kind == 'create' and tid == identity for kind, tid in events) == 1


def test_vector_vjp_matches_sum_backward_for_full_mathematical_loss():
    import torch
    from solver.target_profiles import canonical_target_content_profiles
    from solver.torch_search import BatchedPQCP, TorchSearchConfig
    model = BatchedPQCP(canonical_target_content_profiles(86), torch.device('cpu'), continuous_kernel='dft')
    values = torch.randn(len(model.profiles), 2, 86, requires_grad=True)
    config = TorchSearchConfig(86, continuous_kernel='dft', mathematical_loss='lattice')
    losses = model.loss(values, config)
    summed = torch.autograd.grad(losses.sum(), values, retain_graph=True)[0]
    vector = torch.autograd.grad(losses, values, grad_outputs=torch.ones_like(losses))[0]
    assert torch.equal(summed, vector)


def test_reused_gradient_is_immutable_and_resume_preserves_it(tmp_path):
    import torch
    from solver.torch_search import TorchSearch, TorchSearchConfig
    search = TorchSearch(TorchSearchConfig(14, device='cpu', batch_size=4), root=tmp_path)
    for _ in range(4): search.step()
    assert torch.equal(search._loss_gradient, torch.ones(4))
    search.checkpoint(tmp_path/'state.json')
    restored = TorchSearch.resume(tmp_path/'state.json', root=tmp_path)
    restored.step(); search.step()
    assert torch.equal(restored.theta, search.theta)
    assert torch.equal(restored._loss_gradient, torch.ones(4))
