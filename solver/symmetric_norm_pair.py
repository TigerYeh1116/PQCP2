"""Complete mod4/support certificate for TWO reflection-symmetric trit rows.

This is a restricted mathematical model, never a filter on arbitrary signs
merely because their unsigned supports are symmetric. Independent rotations
center both reflections; a common unit normalizes the nonzero target shift.
"""
from itertools import product

from .cyclic_support import poly_mul,poly_divmod
from .structured_energy import integer_periodic_autocorrelation_profile as pacf


def _linear_preimages(columns,wanted):
    basis={};kernel=[]
    for index,value in enumerate(columns):
        bits=1<<index
        while value:
            high=value.bit_length()-1
            if high not in basis:basis[high]=(value,bits);break
            pivot,toggle=basis[high];value^=pivot;bits^=toggle
        if not value:kernel.append(bits)
    value=wanted;base=0
    while value:
        high=value.bit_length()-1
        if high not in basis:
            check=1<<high
            for pivot in sorted(basis):
                if pivot>high and bin(check&basis[pivot][0]).count('1')%2:check^=1<<pivot
            if (any(bin(check&column).count('1')%2 for column in columns)
                    or bin(check&wanted).count('1')%2!=1):raise RuntimeError('linear inconsistency parity certificate failed')
            return dict(consistent=False,rank=len(basis),base=None,kernel=tuple(kernel),models=(),contradiction_mask=check)
        pivot,toggle=basis[high];value^=pivot;base^=toggle
    models=[]
    for selector in range(1<<len(kernel)):
        bits=base
        for j,toggle in enumerate(kernel):
            if selector>>j&1:bits^=toggle
        image=0
        for j,column in enumerate(columns):
            if bits>>j&1:image^=column
        if image!=wanted:raise RuntimeError('linear reflection support witness failed original equation')
        models.append(bits)
    return dict(consistent=True,rank=len(basis),base=base,kernel=tuple(kernel),models=tuple(models))


def symmetric_norm_support_pairs(d,eta):
    """All mod4-admissible support pairs in the normalized symmetric model.

    N_a+N_b=d delta0+eta(delta1+delta-1), total support weight=d.
    Signed reflection symmetry makes each norm mod4 depend on support only.
    These are necessary supports, not signed/full PQCP witnesses.
    """
    if d not in (3,5,11,43) or type(eta) is not int or eta not in (-1,1):raise ValueError('supported prime and exact target sign required')
    h=(d+1)//2;special=(1<<h)|(1<<(d-h));difference=1|special;modulus=(1<<d)|1
    mul=lambda a,b:poly_divmod(poly_mul(a,b),modulus)[1]
    pairs=tuple((1<<j)|(1<<(d-j)) for j in range(1,d//2+1) if j!=min(h,d-h))
    columns=tuple(mul(p,p)^mul(p,difference) for p in pairs)
    target=(d,eta)+(0,)*(d-3)+(eta,);needed_common_weight=(d-3)//2
    rows=[];retained=[]
    for central,pair in product((0,1),repeat=2):
        left=central|(special if pair else 0);right=difference^left
        norms=tuple(pacf(tuple(mask>>j&1 for j in range(d))) for mask in (left,right))
        residue=tuple((v-x-y)%4 for v,x,y in zip(target,*norms))
        if any(v%2 for v in residue):raise RuntimeError('normalized support square identity failed')
        rhs=sum((v//2)<<j for j,v in enumerate(residue));plan=_linear_preimages(columns,rhs);candidates=[]
        for bits in plan['models']:
            common=0
            for j,p in enumerate(pairs):
                if bits>>j&1:common^=p
            masks=common|left,common|right
            literal=tuple(x+y for x,y in zip(*(pacf(tuple(m>>j&1 for j in range(d))) for m in masks)))
            if any((x-y)%4 for x,y in zip(literal,target)):raise RuntimeError('reflection support model failed literal full mod4 norm audit')
            weight=bin(common).count('1');candidates.append(dict(common_mask=common,common_weight=weight,masks=masks))
            if weight==needed_common_weight:retained.append(masks)
        rows.append(dict(central=central,special_pair=pair,exclusive_masks=(left,right),rhs=rhs,
                         affine=plan,candidates=tuple(candidates)))
    return dict(d=d,eta=eta,complete=True,target=target,difference_support=difference,
        common_pair_basis=pairs,linear_columns=columns,required_common_weight=needed_common_weight,
        rows=tuple(rows),support_pairs=tuple(retained),
        excludes_entire_symmetric_model=not retained,
        scope='TWO signed trit rows individually reflection-symmetric, total weight d. All reflection centers and nonzero target shifts by exact gauges. Arbitrary signs on symmetric supports are NOT excluded.')
