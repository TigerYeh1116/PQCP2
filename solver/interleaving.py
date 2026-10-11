"""Integer identities for interleaving two pairs of odd-length sign rows.

These are sufficient constructions, never exclusions of the full PQCP space.
All equal correlation profiles retain every offset/preimage.
"""
from collections import defaultdict


def periodic_sign_profile(row):
    if not row or any(v not in (-1, 1) for v in row):
        raise ValueError('nonempty sign row required')
    d = len(row)
    return tuple(sum(row[i]*row[(i+u) % d] for i in range(d)) for u in range(d))


def interleave_bits(e, o, offset=0, odd_sign=1):
    if len(e) != len(o) or not e or odd_sign not in (-1, 1):
        raise ValueError('equal nonempty rows and a sign required')
    if any(v not in (-1, 1) for row in (e, o) for v in row):
        raise ValueError('sign coordinates required')
    d = len(e)
    return tuple((1-v)//2 for i in range(d) for v in (e[i], odd_sign*o[(i+offset) % d]))


def odd_cross_vectors(e, o):
    """Map the independent odd PACF coordinates to ALL relative offsets.

    CC(t)=sum_i e[i+t] o[i]. For x=interleave(e,o shifted by delta),
    P_x(2r+1)=CC(r+1-delta)+CC(-r-delta). For odd d, r=(d-1)/2
    is the self-reverse half shift; all other odd shifts occur in pairs.
    """
    if len(e) != len(o) or not e or len(e) % 2 != 1:
        raise ValueError('equal odd length rows required')
    if any(v not in (-1, 1) for row in (e, o) for v in row):
        raise ValueError('sign coordinates required')
    d = len(e)
    cross = tuple(sum(e[(i+t) % d]*o[i] for i in range(d)) for t in range(d))
    groups = defaultdict(list)
    for delta in range(d):
        vector = tuple(cross[(r+1-delta) % d]+cross[(-r-delta) % d] for r in range(d//2+1))
        groups[vector].append(delta)
    return dict(groups)


def sparse_odd_matches(left, right, eta=1):
    """Exact dictionary join H_left-H_right=4*eta at ONE non-half shift.

    Tuple equality is exact even when dictionary hashes collide. Every input
    preimage is retained. The half shift is required to be zero.
    """
    if eta not in (-1, 1):
        raise ValueError('target sign required')
    for vector, left_preimages in left.items():
        for r in range(len(vector)-1):
            required = vector[:r]+(vector[r]-4*eta,)+vector[r+1:]
            for first in left_preimages:
                for second in right.get(required, ()):
                    yield 2*r+1, first, second


def best_sparse_odd_candidate(left,right,eta=1):
    """Exact best energy among supplied odd vectors with zero midpoint.

    This selects one heuristic non-solution seed; it does not exclude omitted
    preimages from existence search. All vectors/target positions participate.
    Even sidelobes must separately be proved zero by the caller.
    """
    if eta not in (-1,1):raise ValueError('target sign required')
    best=None
    for a,pa in left.items():
        for b,pb in right.items():
            if len(a)!=len(b) or len(a)<2 or not pa or not pb:
                raise ValueError('equal nonempty independent odd profiles and preimages required')
            if a[-1]!=b[-1]:continue
            delta=tuple(x-y for x,y in zip(a[:-1],b[:-1]))
            r=max(range(len(delta)),key=lambda i:eta*delta[i])
            residual=sum(v*v for v in delta)-8*eta*delta[r]+16
            if residual%16:raise ValueError('odd profile residual outside binary pair lattice')
            if best is None or residual<best['residual_squared']:
                best=dict(residual_squared=residual,full_energy=residual//16,k=2*r+1,eta=eta,
                          left_preimage=pa[0],right_preimage=pb[0],difference=delta+(0,))
    return best
