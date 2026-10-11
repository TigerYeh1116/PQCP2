from itertools import combinations,product

from solver.support_cut_projection import attainable_cuts,support_cut_obstruction,verify_cut_certificate
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


def test_every_small_literal_cardinality_cut_projection_matches_enumeration():
    for d in (3,5,11):
        # Entire support domain at 3/5; deterministic path/cycle cases at 11.
        masks=range(1<<d) if d<11 else (0,1,(1<<d)-1,0b10110101101,0b11110001111)
        for mask in masks:
            vertices=[j for j in range(d) if mask&(1<<j)]
            for count in range(len(vertices)+1):
                for shift in range(1,d):
                    expected=0
                    for chosen in combinations(vertices,count):
                        minus=set(chosen)
                        cuts=sum(bool(mask&(1<<((j+shift)%d))) and ((j in minus)!=((j+shift)%d in minus)) for j in vertices)
                        expected|=1<<cuts
                    assert attainable_cuts(mask,d,count,shift)==expected


def test_single_and_two_unknown_norms_never_exclude_small_literal_models():
    rows=tuple(product((-1,0,1),repeat=3))
    for a,b in product(rows,repeat=2):
        masks=tuple(sum((v!=0)<<j for j,v in enumerate(r)) for r in (a,b))
        negative=tuple(r.count(-1) for r in (a,b))
        wanted=tuple(x+y for x,y in zip(pacf(a),pacf(b)))
        assert support_cut_obstruction(masks,3,wanted,negative) is None
        assert support_cut_obstruction(masks[:1],3,pacf(a),negative[:1]) is None


def test_fixed_cardinality_cuts_strictly_strengthen_global_degree_bound():
    # Two length-two path components and an isolated vertex, m=1.
    # Every negative active vertex cuts <=1 edge (old global bound is 2).
    mask=0b1101101;d=11;negative=1;shift=1
    intersection=bin(mask&(mask<<shift)).count("1")
    wanted=[0]*d;wanted[0]=bin(mask).count("1");wanted[shift]=wanted[-shift]=intersection-4
    assert abs(wanted[shift])<=intersection and intersection-wanted[shift]<=4*negative
    proof=support_cut_obstruction((mask,),d,wanted,(negative,))
    assert proof is not None and verify_cut_certificate(proof)
    assert proof['shift']==1
    bad=dict(proof,attainable_cut_sums=proof['attainable_cut_sums']^1)
    assert not verify_cut_certificate(bad)


def test_two_unknown_norms_add_all_possible_cut_counts():
    masks=(0b1101101,0b1101101);d=11;negative=(1,1)
    intersection=2*bin(masks[0]&(masks[0]<<1)).count("1")
    wanted=[0]*d;wanted[0]=sum(bin(m).count("1") for m in masks);wanted[1]=wanted[-1]=intersection-6
    proof=support_cut_obstruction(masks,d,wanted,negative)
    assert proof is not None and proof['shift']==1 and verify_cut_certificate(proof)


def test_large_prime_positive_controls_at_every_shift():
    for mask in ((1<<43)-1,0x5F1C37B5A21,0x32494D32295):
        mask&=(1<<43)-1
        row=tuple((-1 if j%3 else 1) if mask&(1<<j) else 0 for j in range(43))
        assert support_cut_obstruction((mask,),43,pacf(row),(row.count(-1),)) is None
