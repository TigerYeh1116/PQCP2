from itertools import product

from experiments.audit_half_constraint_redundancy import constraint_key
from solver.solution_equivalence import _row_key
from solver.verifier import verify_pqcp


def test_constraint_orbits_preserve_empty_vs_nonempty_for_entire_l6_space():
    rows=tuple(product((0,1),repeat=6));valid={}
    for a,b in product(rows,repeat=2):
        if not verify_pqcp(a,b).is_valid:continue
        halves=a[::2],a[1::2],b[::2],b[1::2]
        for free in range(4):valid.setdefault((free,tuple(r for i,r in enumerate(halves) if i!=free)),set()).add((a,b))
    small=tuple(product((0,1),repeat=3));statuses={};positive=negative=collisions=0
    for free in range(4):
        for fixed in product(small,repeat=3):
            halves=list(fixed);halves.insert(free,None);key=constraint_key(halves,free)
            status=bool(valid.get((free,fixed),()))
            positive+=status;negative+=not status
            if key in statuses:
                assert statuses[key]==status;collisions+=1
            statuses[key]=status
    assert positive and negative and collisions


def test_homometric_known_full_rows_share_constraint_key_without_row_equivalence():
    d=11;p=(0,1,3);q=(0,7,13)
    a={((x+y)%(2*d)) for x in p for y in q}
    b={((x-y)%(2*d)) for x in p for y in q}
    assert len(a)==len(b)==9
    aa=tuple(int(i in a) for i in range(2*d));bb=tuple(int(i in b) for i in range(2*d))
    assert _row_key(''.join(map(str,aa)))!=_row_key(''.join(map(str,bb)))
    companion=(0,1,0,1,1,0,0,1,0,0,0)
    assert constraint_key((aa[::2],aa[1::2],companion,None),3)==constraint_key((bb[::2],bb[1::2],companion,None),3)
