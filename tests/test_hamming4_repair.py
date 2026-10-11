from itertools import product
import ctypes
import random

from solver.correlation import full_correlation_profile
from solver.hamming4_repair import _library, exact_hamming4_repair, row_four_flip_search
from solver.interleaving import periodic_sign_profile
from solver.structured_energy import target_profile
from solver.target_profiles import pair_content,target_content_profiles


def test_native_single_row_four_flips_match_complete_small_binary_space():
    rows = tuple(product((0,1),repeat=6))
    other = (0,0,0,0,1,1)
    for row in rows:
        content = pair_content(row,other)
        targets = tuple(t for t in target_content_profiles(6) if
            (t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones)==content)
        desired = {target_profile(6,t.k,t.eta) for t in targets}
        possible = [r for r in rows if sum(a!=b for a,b in zip(row,r))==4
            and pair_content(r,other)==content and tuple(full_correlation_profile(r,other)) in desired]
        out = row_four_flip_search(row,other,targets)
        assert (out['word'] is not None)==bool(possible)
        if possible: assert out['word'] in possible


def test_radius_four_wrapper_matches_literal_small_fixed_content_ball():
    rows = tuple(product((0,1),repeat=6))
    for center in (((0,0,0,0,1,1),(0,0,1,1,0,1)), ((0,1,0,1,1,0),(1,0,1,0,0,1))):
        content = pair_content(*center)
        desired = {target_profile(6,t.k,t.eta) for t in target_content_profiles(6) if
            (t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones)==content}
        possible = [(a,b) for a,b in product(rows,repeat=2) if pair_content(a,b)==content
            and sum(x!=y for old,new in zip(center,(a,b)) for x,y in zip(old,new))<=4
            and tuple(full_correlation_profile(a,b)) in desired]
        out = exact_hamming4_repair(*center)
        assert (out['status']=='SAT')==bool(possible)
        if possible: assert all(w['words'] in possible for w in out['witnesses'])


def test_native_128_bit_rotation_matches_full_integer_pacf_across_word_boundary():
    function = _library().pqcp_binary_profile128
    function.argtypes = [ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    rng = random.Random(1436)
    for length in (6,64,66,86,94):
        masks = [0,(1<<length)-1,1<<(length-1),1<<min(63,length-1)]
        masks += [rng.getrandbits(length) for _ in range(16)]
        for mask in masks:
            output = (ctypes.c_int*length)()
            assert function(mask & ((1<<64)-1),mask>>64,length,output)==0
            signs = tuple(1-2*((mask>>i)&1) for i in range(length))
            assert tuple(output)==periodic_sign_profile(signs)
