from collections import defaultdict
from itertools import product

from solver.norm_trace_completion import norm_trace_preimages,cyclic_cross_trace,_cyclic_cross_trace_polynomial
from solver.sds_norm_completion import cyclic_binary_norm


def symmetric_targets(d,origin=True):
    for half in product((0,1),repeat=d//2):
        for zero in (0,1) if origin else (0,):
            yield zero+sum(v*((1<<i)|(1<<(d-i))) for i,v in enumerate(half,1))


def test_every_small_partner_norm_trace_fiber_against_entire_literal_space():
    for d in (3,5):
        for partner in range(1<<d):
            expected=defaultdict(set)
            for mask in range(1<<d):
                trace=_cyclic_cross_trace_polynomial(mask,partner,d)
                assert cyclic_cross_trace(mask,partner,d)==trace
                expected[(cyclic_binary_norm(mask,d),trace)].add(mask)
            for n,t in product(symmetric_targets(d),symmetric_targets(d,False)):
                out=norm_trace_preimages(n,t,partner,d)
                assert out['complete'] and set(out['masks'])==expected.get((n,t),set())
                reference=norm_trace_preimages(n,t,partner,d,native=False)
                assert set(reference['masks'])==set(out['masks'])


def test_l11_all_reachable_fibers_and_zero_partner_fallback():
    d=11
    for partner in (0,1,(1<<d)-1,0b10101100101):
        expected=defaultdict(set)
        for mask in range(1<<d):expected[(cyclic_binary_norm(mask,d),cyclic_cross_trace(mask,partner,d))].add(mask)
        for (n,t),masks in expected.items():
            out=norm_trace_preimages(n,t,partner,d)
            assert out['complete'] and set(out['masks'])==masks
            assert set(norm_trace_preimages(n,t,partner,d,native=False)['masks'])==masks


def test_l43_nonempty_generic_fiber_has_at_most_eight_literal_preimages():
    mask=1687788056856;partner=3493597975730
    n=cyclic_binary_norm(mask,43);t=cyclic_cross_trace(mask,partner,43)
    out=norm_trace_preimages(n,t,partner)
    assert out['complete'] and mask in out['masks'] and len(out['masks'])<=8
    assert out['fallback_components']==0 and not out['rotation_quotient']
    assert set(norm_trace_preimages(n,t,partner,native=False)['masks'])==set(out['masks'])


def test_native_all_l43_field_targets_against_every_field_element():
    from solver.cyclic_support import support_ring,field_mul,field_pow
    from solver.norm_trace_native import field_norm_trace_roots
    for factor in support_ring(43).factors[1:]:
        degree=14;q=128
        columns=tuple(field_pow(1<<i,q,factor) for i in range(degree))
        stars=[0]*(1<<degree)
        for u in range(1,1<<degree):
            bit=u&-u;stars[u]=stars[u^bit]^columns[bit.bit_length()-1]
        fixed=tuple(u for u in range(1<<degree) if stars[u]==u)
        assert len(fixed)==q
        for w in (1,0b10101100101101):
            expected=defaultdict(set)
            for u in range(1<<degree):
                n=field_mul(u,stars[u],factor)
                h=field_mul(w,stars[u],factor)^field_mul(stars[w],u,factor)
                expected[(n,h)].add(u)
            for n,h in product(fixed,repeat=2):
                actual=field_norm_trace_roots(n,h,w,factor)
                assert set(actual)==expected.get((n,h),set())
