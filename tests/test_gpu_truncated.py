"""The GPU truncated scan must report exactly the CPU truncated_scan witnesses (d=23)."""
import random

import pytest

from solver.compressed_repair import _validated_reference
from solver.truncated_search import family_supports, family_target, top_signs, truncated_scan


def test_gpu_truncated_matches_cpu_on_l46():
    from solver.gpu_truncated import GpuTruncated, pack_job
    try:
        gpu = GpuTruncated(23)
    except Exception as error:
        pytest.skip('no Metal GPU: {}'.format(error))
    import experiments.search_l94_supports as search
    search._GROUPS = None
    groups = search._groups(tuple(range(1, 23)), 46)
    search._GROUPS = None
    d, rng, jobs, cpu = 23, random.Random(5), [], []
    while len(jobs) < 300:
        group = groups[len(jobs) % len(groups)]
        (an, az, ap), (bn, bz, bp) = group['contents']
        a, b, _, wanted = _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])
        k = min(group['targets'][0].k, d-group['targets'][0].k)
        support_a = sum(1 << j for j in rng.sample(range(d), an+ap))
        if not family_target(d, support_a, k)[0]:
            continue
        patterns = top_signs(d, support_a, an, 100)[0]
        cpu.append(sorted(truncated_scan(d, support_a, k, d-an-ap, bn, wanted, patterns)[0]))
        jobs.append(pack_job(d, family_supports(d, support_a, k, d-an-ap, raw=True), wanted, bn, patterns))
    hits, wide = gpu.scan(jobs)
    assert wide == 0
    assert [sorted(h) for h in hits] == cpu
    assert sum(map(len, cpu)) >= 100  # The sample must contain witnesses, not only misses.
