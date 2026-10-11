"""Exact global twice-prime PQCP constraints; no fixed compressed row.

PAF_X(u)=2d-2*number of XOR mismatches. Necessary C/G character
equations and row symmetries yield canonical half-row contents. No
timeout is UNSAT, and every SAT model needs the original integer verifier.
"""
from math import isqrt

SUPPORTED_D=(3,5,11,43)


def content_cases(d,k,eta):
    if (d not in SUPPORTED_D or type(k) is not int or not 1<=k<d
            or type(eta) is not int or eta not in (-1,1)):
        raise ValueError('supported twice-prime strict target required')
    c_total=d+2*eta;g_total=d+2*eta*(-1 if k%2 else 1);cases=[]
    for a in range(isqrt(c_total)+1):
        b=isqrt(c_total-a*a)
        if b*b!=c_total-a*a or a>b:continue
        for g in range(isqrt(g_total)+1):
            h=isqrt(g_total-g*g)
            if h*h!=g_total-g*g or (a+g)%2!=1 or (b+h)%2!=1:continue
            sums=(a+g,a-g,b+h,b-h)
            if any(abs(x)>d for x in sums):continue
            cases.append(dict(d=d,k=k,eta=eta,c_sums=(a,b),g_sums=(g,h),
                half_sums=sums,negative_half_counts=tuple((d-x)//2 for x in sums)))
    return tuple(cases)


def normalized_cases(d=43):
    """Odd negative targets alternate to positive; common units fix k=1/2."""
    return tuple(case for k,eta in ((1,1),(2,1),(2,-1)) for case in content_cases(d,k,eta))


def build_global_solver(case,*,character_pruning=True,seed=0,encoding=None):
    import z3
    d=case['d'];length=2*d;k=case['k'];eta=case['eta']
    if case not in content_cases(d,k,eta) or type(character_pruning) is not bool:
        raise ValueError('derived canonical content case and boolean pruning switch required')
    encoding=encoding or ('intersection' if character_pruning else 'xor')
    if encoding not in ('intersection','xor'):raise ValueError('exact XOR or intersection encoding required')
    solver=z3.SolverFor('QF_FD');solver.set(random_seed=seed)
    rows=tuple(tuple(z3.Bool('x_{}_{}'.format(side,i)) for i in range(length)) for side in range(2))
    # Full row sums are canonicalized by row negation and row exchange.
    # The even part has positive sum in every canonical C/G content case,
    # so an independent even rotation can place a positive bit at zero.
    for side,row in enumerate(rows):
        solver.add(z3.Not(row[0]))
        solver.add(z3.PbEq([(v,1) for v in row],d-case['c_sums'][side]))
    if character_pruning:
        for side,row in enumerate(rows):
            for parity in (0,1):
                count=case['negative_half_counts'][2*side+parity]
                solver.add(z3.PbEq([(row[i],1) for i in range(parity,length,2)],count))
            # Even and chosen odd majority sets both have >d/2 members;
            # their intersection is nonempty. This fixes one even rotation.
            solver.add(rows[side][1] == (case['half_sums'][2*side+1]<0))
    for shift in range(1,d+1):
        if encoding=='intersection':
            value=d-sum(case['c_sums'])
            if shift==d:
                terms=[z3.And(row[i],row[i+d]) for row in rows for i in range(d)]
                if value%2:raise RuntimeError('canonical character parity failed')
                required=value//2
            else:
                terms=[z3.And(row[i],row[(i+shift)%length]) for row in rows for i in range(length)]
                required=value+eta*int(shift==k)
            solver.add(z3.PbEq([(v,1) for v in terms],required))
        elif shift==d:
            mismatches=[z3.Xor(row[i],row[i+d]) for row in rows for i in range(d)]
            required=d  # Each antipodal edge occurs twice in full PAF.
        else:
            mismatches=[z3.Xor(row[i],row[(i+shift)%length]) for row in rows for i in range(length)]
            required=length-2*eta*int(shift==k)
        if encoding=='xor':solver.add(z3.PbEq([(v,1) for v in mismatches],required))
    return solver,rows


def canonicalize_pair(words):
    """Diagnostic normalization used to audit global coverage on small words."""
    from .verifier import verify_pqcp
    words=tuple(tuple(row) for row in words);check=verify_pqcp(*words)
    if not check.is_valid:raise ValueError('verified strict binary PQCP required')
    length=len(words[0]);d=length//2;k=check.nonzero_shifts[0];eta=check.profile[k]//4
    unit=k if k%2 else k//2
    if not unit%2:unit+=d
    words=tuple(tuple(row[unit*i%length] for i in range(length)) for row in words)
    k=1 if k%2 else 2
    if k%2 and eta<0:
        words=tuple(tuple(v^(i%2) for i,v in enumerate(row)) for row in words);eta=1
    gauged=[]
    for row in words:
        if sum(1-2*v for v in row)<0:row=tuple(1-v for v in row)
        if sum((1 if i%2==0 else -1)*(1-2*v) for i,v in enumerate(row))<0:
            row=row[d:]+row[:d]
        gauged.append(row)
    gauged.sort(key=lambda r:sum(1-2*v for v in r))
    counts=tuple(sum(row[i] for i in range(parity,length,2)) for row in gauged for parity in (0,1))
    case=next(c for c in content_cases(d,k,eta) if c['negative_half_counts']==counts)
    result=[]
    for side,row in enumerate(gauged):
        odd_bit=int(case['half_sums'][2*side+1]<0)
        offset=next(2*j for j in range(d) if row[2*j]==0 and row[2*j+1]==odd_bit)
        result.append(row[offset:]+row[:offset])
    check=verify_pqcp(*result)
    if not check.is_valid or check.nonzero_shifts!=(k,length-k) or check.profile[k]!=4*eta:
        raise RuntimeError('canonicalization failed independent original verification')
    return case,tuple(result)
