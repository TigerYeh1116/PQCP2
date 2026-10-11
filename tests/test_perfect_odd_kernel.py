from itertools import product

from experiments.search_perfect43_basis import perfect43_bases
from solver.perfect_odd_kernel import interleave_from_complementary_trits,perfect43_kernel_certificate,rank_mod_prime
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile
from solver.correlation import full_correlation_profile
from solver.verifier import verify_pqcp


def test_perfect43_certificates_verify_gram_ranks_and_every_antisymmetric_convolution():
    for ingredient in perfect43_bases()['rows']:
        certificate=perfect43_kernel_certificate(ingredient['signs'])
        assert certificate['primitive'] and certificate['ranks']=={2:21,11:21}


def test_small_integral_kernel_classification_matches_all_literal_trit_vectors():
    # At d=3 the Gram is 2I and mod-2 rank is one, so the same primitive
    # basis argument can be checked against the ENTIRE trit space literally.
    o=(1,1,-1)
    g=tuple((o[(i-1)%3]-o[(i+1)%3])//2 for i in range(3))
    assert sum(v*v for v in g)==2 and rank_mod_prime((g,),2)==1
    actual=set()
    for word in product((-1,0,1),repeat=3):
        cross=tuple(sum(word[(i+u)%3]*o[i] for i in range(3)) for u in range(3))
        if all(cross[u]+cross[(-u)%3]==0 for u in range(3)):actual.add(word)
    assert actual=={(0,0,0),g,tuple(-v for v in g)}


def test_single_basis_small_construction_has_exact_full_project_target():
    o=(1,1,-1)
    g=tuple((o[(i-1)%3]-o[(i+1)%3])//2 for i in range(3))
    c=tuple(int(v==0) for v in g)
    assert pacf(c)==(1,0,0)
    words=interleave_from_complementary_trits(c,g,o)
    assert verify_pqcp(*words).is_valid
    assert tuple(full_correlation_profile(*words))==target_profile(6,2,-1)
