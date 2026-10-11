from experiments import benchmark_weight_group_pipeline as pipeline
from experiments.run_l86_weight_group_search import bounded_trials, coarse_carriers
from solver.projection_group_targets import group_projection_targets
from solver.structured_energy import fold_profile, target_profile, integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import canonical_target_content_profiles
from solver.torch_compressed_search import paired_contents
from solver.weight_group_targets import group_weight_targets


def test_every_exported_carrier_retains_its_own_target_content_and_integer_norm():
    choices = tuple((t, c) for t in canonical_target_content_profiles(6) for c in paired_contents(t))
    groups = group_weight_targets(group_projection_targets(choices))
    pipeline.initialize_worker(groups)
    positive = False
    for sharing in (False, True):
        for number in range(64):
            row = pipeline.trial((number, 123, 2**23, 'candidate' if sharing else 'rollback'))
            for carrier in coarse_carriers(row, groups[row['group']], sharing):
                positive = True
                a, b = carrier['result']['attempts']['coarse']['words']
                t = carrier['target']
                expected = tuple(v//4 for v in fold_profile(target_profile(t['L'], t['k'], t['eta']), 2))
                assert tuple(x+y for x, y in zip(pacf(a), pacf(b))) == expected
                assert tuple((w.count(-1), w.count(0), w.count(1)) for w in (a, b)) == carrier['contents']
                assert not carrier['result']['verified']
    assert positive


def test_ordered_queue_has_bounded_work_in_flight(monkeypatch):
    monkeypatch.setattr(pipeline, 'trial', lambda job: job)

    class Pool:
        pending = peak = 0

        def submit(self, fn, jobs):
            self.pending += 1
            self.peak = max(self.peak, self.pending)
            result = fn(jobs)
            pool = self

            class Future:
                def result(self):
                    pool.pending -= 1
                    return result
            return Future()

    pool = Pool()
    assert list(bounded_trials(pool, iter(range(257)), 2)) == list(range(257))
    assert pool.peak <= 4 and pool.pending == 0
