"""Exact affine odd-interleave kernels; restricted construction, not a global filter.

For a sum-one perfect odd binary O, let m=(n+1)/2 and S have row O and
rows (O[i-t]+O[i+t])/2. S*g=e_t imposes one positive odd sidelobe pair.
An audited integer right inverse supplies a particular solution. The entire
integer fiber is enumerated in orthogonal primitive kernel coordinates,
using the exact identity m*||g||^2 = 3 + sum(r_s^2), r_s=q_s mod m.
"""
from math import isqrt

from .interleaving import periodic_sign_profile
from .perfect_odd_kernel import rank_mod_prime


def _prime_divisors(value):
    primes=[]
    p=2
    while p*p<=value:
        if value%p==0:
            primes.append(p)
            while value%p==0:value//=p
        p+=1
    if value>1:primes.append(value)
    return tuple(primes)


def perfect_affine_kernel_certificate(o):
    """Fail closed unless all Gram, saturation and integer inverse audits pass."""
    from sympy import Matrix, ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    o=tuple(o)
    n=len(o)
    if (n<3 or n%4!=3 or any(type(v) is not int or v not in (-1,1) for v in o)
            or sum(o)!=1 or periodic_sign_profile(o)!=(n,)+(-1,)*(n-1)):
        raise ValueError('sum-one perfect binary row of length 3 mod 4 required')
    h=n//2;m=(n+1)//2
    basis=tuple(tuple((o[(i-s)%n]-o[(i+s)%n])//2 for i in range(n)) for s in range(1,h+1))
    equations=(o,)+tuple(tuple((o[(i-s)%n]+o[(i+s)%n])//2 for i in range(n)) for s in range(1,h+1))
    if any(sum(x*y for x,y in zip(a,b))!=m*int(i==j)
           for i,a in enumerate(basis) for j,b in enumerate(basis)):
        raise RuntimeError('orthogonal kernel Gram identity failed')
    if any(sum(x*y for x,y in zip(a,b)) for a in equations for b in basis):
        raise RuntimeError('kernel vector failed original linear equations')
    if any(sum(x*y for x,y in zip(a,b))!=(2*m if i==0 else m)*int(i==j)-1
           for i,a in enumerate(equations) for j,b in enumerate(equations)):
        raise RuntimeError('affine equation Gram identity failed')
    ranks={p:rank_mod_prime(basis,p) for p in _prime_divisors(m)}
    if any(rank!=h for rank in ranks.values()):
        raise RuntimeError('integer kernel saturation unproved; do not exclude other vectors')
    matrix=Matrix(equations)
    diagonal,left,right=smith_normal_decomp(DomainMatrix.from_Matrix(matrix).convert_to(ZZ))
    diagonal,left,right=diagonal.to_Matrix(),left.to_Matrix(),right.to_Matrix()
    invariants=tuple(abs(int(diagonal[i,i])) for i in range(h+1))
    if invariants!=(1,)*(h+1):
        raise RuntimeError('affine integer right inverse unavailable')
    inverse=right[:,:h+1]*Matrix.diag(*(diagonal[i,i] for i in range(h+1)))*left
    if any(not v.is_Integer for v in inverse):raise RuntimeError('nonintegral affine inverse')
    inverse=tuple(tuple(int(inverse[i,j]) for j in range(h+1)) for i in range(n))
    # Independent literal-integer certificate: S R = I, not a solver verdict.
    if any(sum(equations[i][k]*inverse[k][j] for k in range(n))!=int(i==j)
           for i in range(h+1) for j in range(h+1)):
        raise RuntimeError('affine integer right-inverse audit failed')
    return dict(o=o,basis=basis,equations=equations,integer_right_inverse=inverse,
                gram_diagonal=m,dimension=h,ranks=ranks,smith_invariants=invariants,
                primitive=True,scope='Identical perfect odd rows; positive odd target only.')


def residue_sphere_vectors(residues, modulus, squared_bound):
    """Enumerate ALL integer vectors r=q mod modulus inside a closed sphere."""
    residues=tuple(residues)
    if (type(modulus) is not int or modulus<1 or type(squared_bound) is not int
            or any(type(v) is not int for v in residues)):
        raise ValueError('integer residues, positive modulus and integer bound required')
    if squared_bound<0:return
    radius=isqrt(squared_bound)
    choices=tuple(tuple(v for v in range(-radius,radius+1) if (v-q)%modulus==0) for q in residues)
    if any(not row for row in choices):return
    suffix=[0]*(len(choices)+1)
    for i in range(len(choices)-1,-1,-1):suffix[i]=suffix[i+1]+min(v*v for v in choices[i])
    word=[0]*len(choices)
    def visit(i,cost):
        if cost+suffix[i]>squared_bound:return
        if i==len(choices):
            yield tuple(word)
            return
        for v in choices[i]:
            word[i]=v
            yield from visit(i+1,cost+v*v)
    yield from visit(0,0)


def perfect_affine_trits(certificate, target, max_norm):
    """Complete norm-bounded integer fiber, retaining every ternary vector."""
    n=len(certificate['o']);h=certificate['dimension'];m=certificate['gram_diagonal']
    if (type(target) is not int or not 1<=target<=h or type(max_norm) is not int
            or not 0<=max_norm<=n):
        raise ValueError('nonzero independent target and norm bound in 0..n required')
    basis=certificate['basis'];equations=certificate['equations']
    particular=tuple(row[target] for row in certificate['integer_right_inverse'])
    q=tuple(sum(x*y for x,y in zip(particular,b)) for b in basis)
    if m*sum(v*v for v in particular)!=3+sum(v*v for v in q):
        raise RuntimeError('affine norm identity failed for particular solution')
    candidates=[];sphere_count=0
    for residues in residue_sphere_vectors(q,m,m*max_norm-3):
        sphere_count+=1
        alpha=tuple((r-v)//m for r,v in zip(residues,q))
        g=tuple(particular[i]+sum(a*b[i] for a,b in zip(alpha,basis)) for i in range(n))
        norm=sum(v*v for v in g)
        if m*norm!=3+sum(v*v for v in residues) or norm>max_norm:
            raise RuntimeError('reconstructed integer vector violated exact sphere identity')
        if any(sum(a*b for a,b in zip(row,g))!=int(i==target) for i,row in enumerate(equations)):
            raise RuntimeError('reconstructed vector failed original affine equations')
        if sum(g)!=2:raise RuntimeError('affine sum identity failed')
        if all(v in (-1,0,1) for v in g):
            candidates.append(dict(g=g,norm=norm,residues=residues,coefficients=alpha))
    return dict(target=target,max_norm=max_norm,particular=particular,q=q,
                sphere_integer_vectors=sphere_count,candidates=candidates,complete=True)
