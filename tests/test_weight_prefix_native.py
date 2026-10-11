"""All projected modular models and original raw-space integer oracles."""
import ctypes
from random import Random

import pytest

from solver.mod8_model_reference import all_mod8_indices
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library, prepare_native_system, run_native_system
from solver.weight_prefix_native import native_arguments, prepare_weight_prefix, run_weight_prefix


def reference_models(prepared):
    p = prepared
    raw = p['plan']
    width, free = int(p['meta'][5]), int(p['meta'][4])
    total = width+free
    u, qword = ctypes.c_uint64, ctypes.c_uint32
    toggles = (u*(2*total))(*p['patterns'][:2*width], *p['free'][:2*free])
    constant = qword()
    linear, quadratic = (qword*total)(), (qword*(total*total))()
    assert _library().pqcp_quadratic_plan(raw['mask_array'], p['base'], toggles, total, raw['d'],
        raw['wanted'], raw['negative'], ctypes.byref(constant), linear, quadratic) == 0
    result = all_mod8_indices((constant.value, tuple(linear), tuple(quadratic)), capacity=1 << total)
    assert result['complete']
    keep = []
    for index in result['indices']:
        pattern = index & ((1 << width)-1)
        if bin(pattern).count('1') != raw['negative'][p['selected']]:
            continue
        if all(bin(pattern & p['checks'][j]).count('1') % 2 == p['rhs'][j] for j in range(p['meta'][6])):
            keep.append(index)
    return tuple(keep)


def model_output(prepared, capacity,count_elimination=False,affine_pruning=False,field_rebasing=False,multi_pivot=False):
    fn = (_library().pqcp_weight_prefix_multi_models if multi_pivot else
          _library().pqcp_weight_prefix_rebased_models if field_rebasing else
          _library().pqcp_weight_prefix_affine_models if affine_pruning else
          _library().pqcp_weight_prefix_count_models if count_elimination else _library().pqcp_weight_prefix_models)
    qp = ctypes.POINTER(ctypes.c_uint32)
    up = ctypes.POINTER(ctypes.c_uint64)
    fn.argtypes = _library().pqcp_weight_signs.argtypes[:-3]+[ctypes.c_uint64, qp, up]+([up] if count_elimination else [])
    fn.restype = ctypes.c_int
    output = (ctypes.c_uint32*capacity)()
    count = ctypes.c_uint64()
    extra=(ctypes.c_uint64*5)()
    args=(*native_arguments(prepared), capacity, output, ctypes.byref(count))
    assert (fn(*args,extra) if count_elimination else fn(*args)) == 0
    return tuple(sorted(output[:min(count.value, capacity)])), count.value


def raw_words(plan, index):
    signs = list(plan['base'])
    for j in range(plan['meta'][2]):
        if index >> j & 1:
            for s in range(2):
                signs[s] ^= plan['toggles'][2*j+s]
    return tuple(tuple(0 if not plan['masks'][s] >> i & 1 else -1 if signs[s] >> i & 1 else 1
                       for i in range(plan['d'])) for s in range(2))


@pytest.mark.parametrize('d', (3, 5, 7, 9, 11))
def test_complete_modular_sets_and_original_integer_solution_sets(d):
    rng = Random(860040+d)
    seen = False
    for _ in range(8):
        active = set(rng.sample(range(2*d), d))
        masks = tuple(sum(1 << i for i in range(d) if s*d+i in active) for s in range(2))
        ref = tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d)) for m in masks)
        wanted = tuple(x+y for x, y in zip(pacf(ref[0]), pacf(ref[1])))
        negative = tuple(w.count(-1) for w in ref)
        raw = prepare_native_system(masks, d, wanted, negative, conditioning=False)
        valid = set()
        for index in range(1 << raw['meta'][2]):
            rows = raw_words(raw, index)
            if tuple(w.count(-1) for w in rows) == negative and tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1]))) == wanted:
                valid.add(rows)
        assert valid
        for side in (0, 1):
            p = prepare_weight_prefix(raw, 2**23, selected=side)
            if p is None:
                continue
            seen = True
            expected = reference_models(p)
            actual, count = model_output(p, 1 << (p['meta'][5]+p['meta'][4]))
            assert actual == expected and count == len(expected)
            affine_actual, affine_count = model_output(p, 1 << (p['meta'][5]+p['meta'][4]), True, True)
            assert affine_actual == expected and affine_count == len(expected)
            rebased_actual, rebased_count = model_output(p, 1 << (p['meta'][5]+p['meta'][4]), True, False, True)
            assert rebased_actual == expected and rebased_count == len(expected)
            multi_actual, multi_count = model_output(p, 1 << (p['meta'][5]+p['meta'][4]), True, False, False, True)
            assert multi_actual == expected and multi_count == len(expected)
            result = run_weight_prefix(raw, 2**23, selected=side)
            assert result['status'] == 'SAT_COMPRESSED' and result['words'] in valid
            affine_result = run_weight_prefix(raw, 2**23, selected=side, count_elimination=True, affine_pruning=True)
            assert affine_result['status'] == 'SAT_COMPRESSED' and affine_result['words'] in valid
            rebased_result = run_weight_prefix(raw, 2**23, selected=side, count_elimination=True, field_rebasing=True)
            assert rebased_result['status'] == 'SAT_COMPRESSED' and rebased_result['words'] in valid
            multi_result = run_weight_prefix(raw, 2**23, selected=side, count_elimination=True, multi_pivot=True)
            assert multi_result['status'] == 'SAT_COMPRESSED' and multi_result['words'] in valid
            if raw['meta'][2]:
                assert run_weight_prefix(raw, (1 << raw['meta'][2])-1, selected=side) is None
    assert seen


def test_zero_dimensions_projection_rejections_and_overflow_counts():
    for masks, wanted, negative, expected in [((0,0), (0,0,0), (0,0), 1),
                                            ((15,0), (4,-1,-1,-1,-1), (2,0), 2)]:
        raw = prepare_native_system(masks, len(wanted), wanted, negative, conditioning=False)
        p = prepare_weight_prefix(raw, 2**23, selected=0)
        assert p is not None
        output, count = model_output(p, 1)
        assert count == expected and len(output) == 1
        result = run_weight_prefix(raw, 2**23, selected=0)
        assert result['status'] == 'SAT_COMPRESSED'
        if masks[0]:
            # Six literal weight-two patterns, only two projected preimages.
            assert result['weight_patterns'] >= result['compatible_patterns']


@pytest.mark.parametrize('seed', (860050, 860051, 860052))
@pytest.mark.parametrize('width', (8,9,12,14,16,17))
@pytest.mark.parametrize('count_elimination,affine_pruning,field_rebasing,multi_pivot',((False,False,False,False),(True,False,False,False),(True,True,False,False),(True,False,True,False),(True,False,False,True)))
def test_large_fixed_weight_positive_controls_and_all_modular_models(seed,width,count_elimination,affine_pruning,field_rebasing,multi_pivot):
    rng = Random(seed)
    d = 43
    masks = tuple(sum(1 << i for i in rng.sample(range(d), w)) for w in (width,43-width))
    ref = tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d)) for m in masks)
    wanted = tuple(x+y for x,y in zip(pacf(ref[0]),pacf(ref[1])))
    negative = tuple(w.count(-1) for w in ref)
    raw = prepare_native_system(masks, d, wanted, negative, conditioning=False)
    p = prepare_weight_prefix(raw, 2**23)
    assert p is not None and p['meta'][4] <= 14
    expected = reference_models(p)
    actual, count = model_output(p, 1 << (p['meta'][5]+p['meta'][4]),count_elimination,affine_pruning,field_rebasing,multi_pivot)
    assert actual == expected and count == len(expected)
    result = run_weight_prefix(raw, 2**23,count_elimination=count_elimination,affine_pruning=affine_pruning,field_rebasing=field_rebasing,multi_pivot=multi_pivot)
    assert result is not None and result['status'] == run_native_system(raw, 2**23)['status'] == 'SAT_COMPRESSED'
    assert tuple(w.count(-1) for w in result['words']) == negative
    assert tuple(x+y for x,y in zip(pacf(result['words'][0]),pacf(result['words'][1]))) == wanted


def test_same_modular_models_do_not_replace_full_integer_norm_checks():
    rng = Random(860053)
    d = 43
    masks = tuple(sum(1 << i for i in rng.sample(range(d), w)) for w in (16, 27))
    ref = tuple(tuple(rng.choice((-1, 1)) if m >> i & 1 else 0 for i in range(d)) for m in masks)
    wanted = tuple(x+y for x,y in zip(pacf(ref[0]),pacf(ref[1])))
    negative = tuple(w.count(-1) for w in ref)
    for u in range(1, d//2+1):
        terms = sum(sum(bool(m >> i & 1 and m >> ((i+u) % d) & 1) for i in range(d)) for m in masks)
        delta = 8 if wanted[u]+8 <= terms else -8 if wanted[u]-8 >= -terms else None
        if delta is None:
            continue
        bad = list(wanted)
        bad[u] += delta
        bad[d-u] += delta
        good = prepare_native_system(masks, d, wanted, negative, conditioning=False)
        fake = prepare_native_system(masks, d, bad, negative, conditioning=False)
        pg = prepare_weight_prefix(good, 2**23)
        pf = prepare_weight_prefix(fake, 2**23)
        assert pg is not None and pf is not None
        assert reference_models(pg) == reference_models(pf)
        assert run_weight_prefix(good, 2**23)['status'] == 'SAT_COMPRESSED'
        # Counts fix both row sums, hence fix sum_u R(u). The changed
        # symmetric target pair breaks that identity by +/-16.
        result = run_weight_prefix(fake, 2**23)
        assert result['status'] == run_native_system(fake, 2**23)['status'] == 'UNSAT_FIXED_SUPPORT'
        assert result['words'] is None
        return
    pytest.fail('no bounded integer negative control found')


def test_dispatch_honors_explicit_controls_and_original_partial_budgets():
    rng=Random(860054);d=43
    masks=tuple(sum(1<<i for i in rng.sample(range(d),w)) for w in (14,29))
    ref=tuple(tuple(rng.choice((-1,1)) if m>>i&1 else 0 for i in range(d)) for m in masks)
    wanted=tuple(x+y for x,y in zip(pacf(ref[0]),pacf(ref[1])))
    p=prepare_native_system(masks,d,wanted,tuple(w.count(-1) for w in ref),conditioning=False)
    new=run_native_system(p,2**23,weight_prefix=True)
    assert new['hybrid_backend']=='exact_weight_prefix' and new['status']=='SAT_COMPRESSED'
    for controls,budget in (({},(1<<p['meta'][2])-1),({'quadratic':False},4096),
                            ({'hybrid_tail':0},4096),({'conditioned_hybrid':False},4096)):
        old=run_native_system(p,budget,weight_prefix=False,**controls)
        candidate=run_native_system(p,budget,weight_prefix=True,**controls)
        assert (old['status'],old['words'],old['assignments'])==(candidate['status'],candidate['words'],candidate['assignments'])
    with pytest.raises(ValueError):run_native_system(p,2**23,weight_prefix=1)


def test_raw_plan_without_tight_counts_can_fall_back_on_conditioning_obstruction():
    # Three distinct difference distances force all signs equal; the odd
    # row's even count parity then fixes all signs positive, contradicting n=2.
    p=prepare_native_system((11,0),7,(3,1,1,1,1,1,1),(2,0),conditioning=False,cardinality_pruning=False)
    assert p['consistent']
    assert prepare_weight_prefix(p,2**23,selected=0) is None
    assert run_native_system(p,2**23)['status']=='UNSAT_FIXED_SUPPORT'


def test_nonaffine_count_polynomial_keeps_its_complete_tail_equation():
    # A deliberately general affine sign family: its two even toggles
    # overlap once, so the count polar coefficient is one, not zero.
    u,i=ctypes.c_uint64,ctypes.c_int
    p=dict(consistent=True,selected=0,masks=(1,7),d=5,base=(u*2)(0,0),
        patterns=(u*2)(1,0),toggles=(u*4)(0,3,0,6),checks=(u*1)(0),rhs=(i*1)(0),
        meta=(i*8)(4,0,3,1,2,1,0,1),wanted=(i*5)(4,0,-1,-1,0),
        negative=(i*2)(1,2),mask_array=(u*2)(1,7),construction_seconds=0.0)
    prepared=prepare_weight_prefix(p,2**23,selected=0)
    assert prepared is not None
    expected=reference_models(prepared)
    actual,count=model_output(prepared,8,count_elimination=True)
    assert actual==expected and count==len(expected)
    result=run_weight_prefix(p,2**23,selected=0,count_elimination=True)
    assert result['status']=='SAT_COMPRESSED' and result['count_eliminated_patterns']==0
