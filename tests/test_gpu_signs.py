"""GPU sign enumeration must give exactly the CPU solver's verdict on every support pair."""
import random

import pytest

from solver.compressed_repair import _validated_reference
from solver.gf2_support_family import gf2_support_ring, sample_support
from solver.ternary_signs_native import prepare_sign_enumeration, solve_sign_system
from solver.ternary_support_search import support_sign_system


def _pairs(length, weights, count, seed):
    import experiments.search_l94_supports as search
    search._GROUPS = None
    groups = search._groups(weights, length)
    search._GROUPS = None
    d, ring, rng, out = length//2, gf2_support_ring(length//2), random.Random(seed), []
    while len(out) < count:
        group = groups[len(out) % len(groups)]
        (an, az, ap), (bn, bz, bp) = group['contents']
        a, b, d, wanted = _validated_reference((-1,)*an+(0,)*az+(1,)*ap, (-1,)*bn+(0,)*bz+(1,)*bp, group['targets'][0])
        u = sum(1 << j for j in rng.sample(range(d), an+ap))
        v = sample_support(ring, u, min(group['targets'][0].k, d-group['targets'][0].k), d-an-ap, rng)
        if v is not None:
            out.append(((u, v), d, wanted, (an, bn)))
    return out


def _cpu(masks, d, wanted, negative):
    system = support_sign_system(masks, d, wanted, negative)
    if not system.consistent:
        return False
    return solve_sign_system(system, masks, d, wanted, negative, 2**25,
                             prepared=prepare_sign_enumeration(system, masks, negative))['status'] == 'SAT_COMPRESSED'


def test_gpu_matches_cpu_on_l62_pairs():
    from solver.gpu_signs import GpuSigns, prepare_pair
    try:
        gpu = GpuSigns(31)
    except Exception as error:
        pytest.skip('no Metal GPU: {}'.format(error))
    cases = _pairs(62, (15, 17), 1200, 7)
    prepared = [prepare_pair(*case) for case in cases]
    live = [i for i, p in enumerate(prepared) if p is not None]
    hits, witnesses = gpu.check([prepared[i] for i in live])
    gpu_verdict = [False]*len(cases)
    for i, hit, witness in zip(live, hits, witnesses):
        gpu_verdict[i] = hit
        if hit:  # The GPU witness must be a genuine negative-sign assignment on the supports.
            assert all(w & ~m == 0 for w, m in zip(witness, cases[i][0]))
    cpu_verdict = [_cpu(*case) for case in cases]
    assert gpu_verdict == cpu_verdict
    assert sum(cpu_verdict) >= 5  # The sample must contain satisfiable pairs, not only misses.
