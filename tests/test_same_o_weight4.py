from itertools import combinations

from solver.same_o_weight4 import c_affine_certificate,iter_c_masks,required_o_intersections,odd_integer_match,normalized_g_rows
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.correlation import full_correlation_profile


def test_all_small_g_and_c_affine_fibers_against_original_integer_mod_four():
    for d in (5,7,9):
        for minus_position in range(d):
            for plus_positions in combinations([i for i in range(d) if i!=minus_position],3):
                plus=sum(1<<i for i in plus_positions);minus=1<<minus_position
                active=tuple(i for i in range(d) if not (plus|minus)>>i&1)
                g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(d));pg=pacf(g)
                for weight in range(len(active)+1):
                    expected=set()
                    for negative_positions in combinations(active,weight):
                        negative=sum(1<<i for i in negative_positions)
                        c=tuple(0 if (plus|minus)>>i&1 else 1-2*int(negative>>i&1) for i in range(d));pc=pacf(c)
                        if all((pc[u]+pg[u]+2-d)%4==0 for u in range(1,d)):expected.add(negative)
                    certificate=c_affine_certificate(plus,minus,d,weight)
                    masks=tuple(iter_c_masks(certificate))
                    assert len(masks)==len(set(masks)) and set(masks)==expected


def test_l43_even_and_odd_equations_are_literal_full_interleave_conditions():
    plus=(1<<1)|(1<<2)|(1<<3);minus=1
    certificate=c_affine_certificate(plus,minus)
    masks=iter_c_masks(certificate)
    negative=next(masks)
    wanted=required_o_intersections(negative,plus,minus)
    assert sum(wanted)==441
    c=tuple(0 if (plus|minus)>>i&1 else 1-2*int(negative>>i&1) for i in range(43))
    g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(43));pc,pg=pacf(c),pacf(g)
    assert sum(c)==9 and sum(g)==2
    for u in range(1,43):assert pc[u]+pg[u]-41+4*wanted[u]==0
    for o_mask in (sum(1<<i for i in range(1,22)),3493597975730):
        o=tuple(1-2*int(o_mask>>i&1) for i in range(43));assert sum(o)==1
        profile=full_correlation_profile(*interleave_from_complementary_trits(c,g,o));target=target_profile(86,41,1)
        assert odd_integer_match(o_mask,plus,minus)==all(profile[u]==target[u] for u in range(1,86,2))


def test_common_reflection_quotient_covers_the_entire_normalized_g_bank():
    from solver.norm_trace_completion import reverse_mask
    all_g=set(normalized_g_rows());representatives={(p,m) for p,m in all_g if p<reverse_mask(p,43)}
    assert len(all_g)==11480 and len(representatives)==5740
    assert representatives|{(reverse_mask(p,43),m) for p,m in representatives}==all_g


def test_native_entire_c_fiber_matches_independent_python_filter_counts():
    from collections import Counter
    from solver.same_o_weight4_native import complete_weight4_c_fiber
    from solver.certified_profile_psd import profile_psd_obstruction
    from solver.norm_trace_completion import norm_trace_preimages
    from solver.sds_norm_completion import intersection_profile
    plus=14;minus=1;certificate=c_affine_certificate(plus,minus)
    native=complete_weight4_c_fiber(certificate);counts=Counter();hits=set()
    assert native['complete'] and native['stats']['affine_assignments']==1<<certificate['nullity']
    for negative in iter_c_masks(certificate):
        counts['weight_matched_c']+=1;wanted=required_o_intersections(negative,plus,minus)
        if any(not 0<=v<=20 for v in wanted[1:]):counts['range_excluded']+=1;continue
        if profile_psd_obstruction(wanted) is not None:counts['psd_excluded']+=1;continue
        norm=sum((v%2)<<i for i,v in enumerate(wanted));o=norm_trace_preimages(norm,(1<<1)|(1<<42),plus|minus,native=False)
        if not o['masks']:counts['norm_trace_empty']+=1;continue
        for mask in o['masks']:
            counts['literal_o_preimages']+=1
            if bin(mask).count('1')!=21:counts['o_weight_excluded']+=1;continue
            if not odd_integer_match(mask,plus,minus):counts['odd_integer_excluded']+=1;continue
            if intersection_profile(mask,43)!=wanted:counts['even_integer_excluded']+=1;continue
            counts['full_hits']+=1;hits.add((negative,mask))
    for key in ('weight_matched_c','range_excluded','psd_excluded','norm_trace_empty','literal_o_preimages',
                'o_weight_excluded','odd_integer_excluded','even_integer_excluded','full_hits'):
        assert native['stats'][key]==counts[key]
    assert hits==set(native['hits']) and counts['literal_o_preimages']>0


def test_nonempty_odd_integer_seed_buffer_retries_and_full_energy_is_exact():
    from solver.same_o_weight4_native import complete_weight4_c_fiber
    from solver.sds_norm_completion import intersection_profile
    from solver.structured_energy import full_error_energy
    plus=1073807362;minus=1;certificate=c_affine_certificate(plus,minus)
    out=complete_weight4_c_fiber(certificate,capacity=1,seed_capacity=1)
    assert out['complete'] and out['stats']['odd_integer_matched']>1
    assert len(out['odd_integer_models'])==out['stats']['odd_integer_matched']
    for negative,o_mask,deviation in out['odd_integer_models']:
        wanted=required_o_intersections(negative,plus,minus);actual=intersection_profile(o_mask,43)
        assert deviation==sum((actual[u]-wanted[u])**2 for u in range(1,22))
        assert odd_integer_match(o_mask,plus,minus)
        c=tuple(0 if (plus|minus)>>i&1 else 1-2*int(negative>>i&1) for i in range(43))
        g=tuple(int(plus>>i&1)-int(minus>>i&1) for i in range(43));o=tuple(1-2*int(o_mask>>i&1) for i in range(43))
        profile=full_correlation_profile(*interleave_from_complementary_trits(c,g,o))
        assert full_error_energy(profile,41,1)==64*deviation


def test_all_small_two_impulse_c_affine_fibers_against_original_mod_four():
    for d in (5,7,9):
        for positions in combinations(range(d),2):
            plus=sum(1<<i for i in positions);active=tuple(i for i in range(d) if i not in positions)
            g=tuple(int(plus>>i&1) for i in range(d));pg=pacf(g)
            for weight in range(len(active)+1):
                expected=set()
                for negative_positions in combinations(active,weight):
                    negative=sum(1<<i for i in negative_positions)
                    c=tuple(0 if plus>>i&1 else 1-2*int(negative>>i&1) for i in range(d));pc=pacf(c)
                    if all((pc[u]+pg[u]+2-d)%4==0 for u in range(1,d)):expected.add(negative)
                certificate=c_affine_certificate(plus,0,d,weight,g_weight=2)
                masks=tuple(iter_c_masks(certificate));assert len(set(masks))==len(masks) and set(masks)==expected


def test_all_l43_impulse_responses_give_exact_full_odd_targets_with_non_skew_o():
    from solver.same_o_weight4 import impulse_symmetry_response
    norms=set()
    for s in range(1,22):
        response=impulse_symmetry_response(s);o=list(response)
        for i in range(1,22):
            if not response[i]:o[i],o[-i]=1,-1
        assert sum(o)==1 and all(v in (-1,1) for v in o)
        assert all(o[i]+o[-i]==2*response[i] for i in range(43))
        plus=(1<<s)|(1<<(43-s));positions=[i for i in range(43) if not plus>>i&1]
        negative=sum(1<<i for i in positions[:16])
        c=tuple(0 if plus>>i&1 else 1-2*int(negative>>i&1) for i in range(43));g=tuple(int(plus>>i&1) for i in range(43))
        profile=full_correlation_profile(*interleave_from_complementary_trits(c,g,o));target=target_profile(86,41,1)
        assert all(profile[u]==target[u] for u in range(1,86,2))
        norms.add(sum(v*v for v in response))
    assert norms==set(range(1,42,2))


def test_two_impulse_native_reproduces_complete_skew_family_integer_counters():
    from solver.same_o_weight4_native import complete_weight2_c_fiber
    from solver.same_o_weight4 import impulse_symmetry_response
    plus=(1<<1)|(1<<42);c=c_affine_certificate(plus,0,negative_weight=16,g_weight=2)
    out=complete_weight2_c_fiber(c)
    assert out['complete'] and out['stats']['weight_matched_c']==122880
    assert out['stats']['psd_excluded']==120882 and out['stats']['norm_trace_empty']==1492
    assert len(out['odd_integer_models'])==4048 and not out['hits']
    response=impulse_symmetry_response(1)
    for negative,o_mask,deviation in out['odd_integer_models']:
        assert odd_integer_match(o_mask,plus,0) and deviation>0
        assert all(1-((o_mask>>i)&1)-((o_mask>>(-i%43))&1)==response[i] for i in range(43))
