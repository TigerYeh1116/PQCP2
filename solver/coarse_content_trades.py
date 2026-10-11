"""Exact content-changing two-site trades between the two coarse C rows.

One row creates +1/-1 at two zeros; the other removes one +1/-1 pair.
Each row sum and total support weight stay fixed. Match the FULL independent
integer PACF deltas to zero, retaining every literal trade/preimage.
"""
from collections import defaultdict
from itertools import combinations

from .compressed_repair import _validated_reference
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .torch_compressed_search import search_paired_contents


def replace_two_delta(row,p,q,vp,vq):
    d=len(row)
    alpha,beta=vp-row[p],vq-row[q]
    return tuple(alpha*(row[(p+u)%d]+row[(p-u)%d])
        +beta*(row[(q+u)%d]+row[(q-u)%d])
        +alpha*beta*(int((p+u)%d==q)+int((q+u)%d==p))
        for u in range(1,d//2+1))


def exact_coarse_content_trades(a,b,target,*,max_results=4096):
    a,b,d,wanted=_validated_reference(a,b,target)
    if type(max_results) is not int or not 1<=max_results<=4096:
        raise ValueError('bounded literal output cap required')
    if tuple(x+y for x,y in zip(pacf(a),pacf(b)))!=wanted:
        raise ValueError('exact coarse witness required')
    allowed=search_paired_contents(target)
    results=[]
    tests=0
    for creator in (0,1):
        pair=(a,b)
        source,donor=pair[creator],pair[1-creator]
        proposed=list((w.count(-1),w.count(0),w.count(1)) for w in pair)
        n,z,p=proposed[creator];proposed[creator]=(n+1,z-2,p+1)
        n,z,p=proposed[1-creator];proposed[1-creator]=(n-1,z+2,p-1)
        if tuple(proposed) not in allowed:continue
        create=defaultdict(list)
        for p,q in combinations([i for i,v in enumerate(source) if v==0],2):
            for sign in (-1,1):
                delta=replace_two_delta(source,p,q,sign,-sign)
                create[delta].append(tuple(sign if i==p else -sign if i==q else v for i,v in enumerate(source)))
        for p in (i for i,v in enumerate(donor) if v==1):
            for q in (i for i,v in enumerate(donor) if v==-1):
                delta=tuple(-v for v in replace_two_delta(donor,p,q,0,0))
                tests+=1
                removed=tuple(0 if i in (p,q) else v for i,v in enumerate(donor))
                for created in create.get(delta,()):
                    words=(created,removed) if creator==0 else (removed,created)
                    if tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))!=wanted:
                        raise RuntimeError('content trade failed independent full coarse integer verifier')
                    if len(results)==max_results:
                        return dict(words=results,complete=False,exact_lookups=tests,status='SAT_TRUNCATED')
                    results.append(words)
    return dict(words=results,complete=True,exact_lookups=tests,
                status='SAT_COMPLETE' if results else 'UNSAT_CONTENT_TRADE_NEIGHBORHOOD')
