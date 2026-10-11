from itertools import product

import pytest

from experiments.search_perfect43_basis import perfect43_bases
from solver.perfect_affine_kernel import (perfect_affine_kernel_certificate,
                                        perfect_affine_trits,residue_sphere_vectors)
from solver.perfect_odd_kernel import interleave_from_complementary_trits
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf,target_profile
from solver.correlation import full_correlation_profile


def test_residue_sphere_complete_against_literal_integer_cube():
    for modulus in (2,3,5):
        for bound in (0,1,4,9):
            residues=(1,-2,0)
            actual=set(residue_sphere_vectors(residues,modulus,bound))
            expected={r for r in product(range(-3,4),repeat=3)
                      if sum(v*v for v in r)<=bound and all((v-q)%modulus==0 for v,q in zip(r,residues))}
            assert actual==expected
    assert list(residue_sphere_vectors((1,),2,-1))==[]


def test_small_affine_fibers_cover_entire_ternary_space_including_nonempty_cases():
    # Lengths 3 and 7: independently enumerate every ternary row, not coordinates.
    for o in ((1,1,-1),(1,1,1,-1,1,-1,-1)):
        certificate=perfect_affine_kernel_certificate(o)
        for target in range(1,len(o)//2+1):
            expected={g for g in product((-1,0,1),repeat=len(o))
                      if all(sum(x*y for x,y in zip(row,g))==int(i==target)
                             for i,row in enumerate(certificate['equations']))}
            assert expected
            for norm in range(len(o)+1):
                out=perfect_affine_trits(certificate,target,norm)
                assert {row['g'] for row in out['candidates']}=={g for g in expected if sum(v*v for v in g)<=norm}


def test_all_perfect43_affine_certificates_have_literal_integer_right_inverse():
    for ingredient in perfect43_bases()['rows']:
        certificate=perfect_affine_kernel_certificate(ingredient['signs'])
        s=certificate['equations'];r=certificate['integer_right_inverse']
        assert all(sum(s[i][k]*r[k][j] for k in range(43))==int(i==j)
                   for i in range(22) for j in range(22))
        out=perfect_affine_trits(certificate,1,34)
        assert out['complete']
        for row in out['candidates']:
            assert sum(row['g'])==2 and row['norm']<=34


def test_affine_even_and_odd_pacf_expansion_on_literal86_words():
    o=(1,)+tuple(1 if pow(i,21,43)==1 else -1 for i in range(1,43))
    certificate=perfect_affine_kernel_certificate(o)
    for t in (1,7,21):
        out=perfect_affine_trits(certificate,t,34)
        assert out['candidates']
        for row in out['candidates'][:3]:
            g=row['g'];c=tuple(int(v==0) for v in g)
            actual=tuple(full_correlation_profile(*interleave_from_complementary_trits(c,g,o)))
            pc=tuple(pacf(c));pg=tuple(pacf(g));po=tuple(pacf(o))
            wanted=target_profile(86,43-2*t,1)
            assert all(actual[2*u]==2*(pc[u]+pg[u]+po[u]) for u in range(43))
            assert all(actual[u]==wanted[u] for u in range(1,86,2))


def test_invalid_perfect_ingredient_is_not_accepted_as_exclusion():
    with pytest.raises(ValueError):perfect_affine_kernel_certificate((1,)*43)
