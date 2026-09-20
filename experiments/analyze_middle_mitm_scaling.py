"""Exact mathematical diagnostics for compressed lifts, not a new solver.

For L=2d, c_i=x_i+x_{i+d}, h_i=(x_i-x_{i+d})/2. Then h is zero
exactly where c is nonzero, otherwise h_i is +/-1. Write N_h for the
NEGACYCLIC autocorrelation (a wrap over d changes sign). Direct expansion
gives P_x(u) = P_c(u)/2 + 2*N_h(u) for 0<=u<d, and
P_x(u+d) = P_c(u)/2 - 2*N_h(u).

Thus, after an exact compressed pair match, the remaining equations are
N_ha(u)+N_hb(u) = (T(u)-T(u+d))/4, NOT a cyclic PACF equation on h.
At u=0, z_a+z_b=d because T(0)=2L and T(d)=0. There are exactly d
binary orientation variables before cross constraints. With L=4n, d even,
each zero contributes one bit=1 to a fixed parity, so parity weights cannot
further reduce these orientations. For n odd, prescribed mod-4 row sums
give exact binomial counts; these count lifts, not solutions to full PACF.

The optional probe only calls the EXISTING binary Z3 solver on one saved
middle problem, in an externally time-bounded process. UNKNOWN remains
unresolved. Nothing here changes production search or its mathematical target.
"""

import argparse
from dataclasses import asdict
import json
from math import comb
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.cross_compression import compatible_cross_compressions, middle_matches_cross
from solver.structured_energy import (
    fold_profile, integer_periodic_autocorrelation_profile as pacf, target_profile,
    compress_binary_sequence,
)
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def negacyclic_profile(word):
    """Exact integer profile with a minus sign for one wrap over len(word)."""
    d = len(word)
    if not d or any(type(x) is not int for x in word):
        raise ValueError('nonempty integer word required')
    return tuple(sum(word[i]*word[(i+u)%d]*(1 if i+u < d else -1)
                     for i in range(d)) for u in range(d))


def split_channels(bits):
    """Return (c, h) from a full even-length Project binary sequence."""
    if not bits or len(bits)%2 or any(type(x) is not int or x not in (0,1) for x in bits):
        raise ValueError('even-length integer bits required')
    d = len(bits)//2
    signs = tuple(1-2*b for b in bits)
    return (tuple(signs[i]+signs[i+d] for i in range(d)),
            tuple((signs[i]-signs[i+d])//2 for i in range(d)))


def residual_target(L, k, eta):
    """The missing anti-periodic channel, using actual target shifts."""
    if L%4:
        raise ValueError('diagnostic requires L divisible by four')
    target = target_profile(L,k,eta)
    d = L//2
    differences = tuple(target[u]-target[u+d] for u in range(d))
    assert all(x%4 == 0 for x in differences)
    return tuple(x//4 for x in differences)


def orientation_parity_system(ma, mb, target):
    """Exact necessary mod-4 equations on the remaining sign choices.

    Each residual equation is sum_e s_e*y_i*y_j = R, with M terms, s_e=+/-1
    known from the negacyclic wrap and y_i=(-1)^b_i. It must have exactly
    (M-R)/2 negative terms. Multiplying all terms yields a LINEAR GF(2)
    equation: XOR of incident b_i with odd multiplicity equals
    ((M-R)/2 mod 2) XOR (number of negative s_e mod 2).
    Gaussian elimination is exact. Inconsistency proves this FIXED middle
    problem impossible. Consistency is necessary, not sufficient for PACF.
    The rank counts independent parity equations, not quadratic equations.
    """
    ma,mb=tuple(ma),tuple(mb)
    d=len(ma)
    if not d or d%2 or len(mb)!=d or len(target)!=d or any(type(x) is not int or x not in (-2,0,2) for x in ma+mb):
        raise ValueError('equal even-length factor-2 words and residual target required')
    variables=[(s,i) for s,w in enumerate((ma,mb)) for i,x in enumerate(w) if x==0]
    ids={v:j for j,v in enumerate(variables)}
    basis={}
    consistent=True
    raw=[]
    for u in range(1,d//2):
        mask=terms=wraps=0
        for s,w in enumerate((ma,mb)):
            for i in range(d):
                j=(i+u)%d
                if w[i]==w[j]==0:
                    terms+=1
                    wraps^=int(i+u>=d)
                    mask^=(1<<ids[s,i])^(1<<ids[s,j])
        R=target[u]
        if type(R) is not int or (terms-R)%2 or abs(R)>terms:
            consistent=False
            break
        rhs=(((terms-R)//2)&1)^wraps
        raw.append(dict(shift=u,mask=mask,rhs=rhs,terms=terms))
        while mask:
            pivot=mask.bit_length()-1
            if pivot not in basis:
                basis[pivot]=(mask,rhs)
                break
            other,bit=basis[pivot]
            mask^=other;rhs^=bit
        if not mask and rhs:
            consistent=False
            break
    return dict(variables=variables, equations=raw,
                basis=[(p,*basis[p]) for p in sorted(basis)], rank=len(basis),
                consistent=consistent,
                affine_assignments=(1<<(len(variables)-len(basis))) if consistent else 0)


def affine_basis(system):
    """Return one solution and a complete nullspace basis for triangular rows."""
    if not system['consistent']:
        raise ValueError('inconsistent affine system')
    def complete(word,homogeneous):
        for pivot,mask,rhs in system['basis']:
            if (bin(word & (mask^(1<<pivot))).count('1')&1) ^ (0 if homogeneous else rhs):
                word^=1<<pivot
        return word
    pivots={p for p,_,_ in system['basis']}
    return complete(0,False),tuple(complete(1<<f,True)
        for f in range(len(system['variables'])) if f not in pivots)


def affine_binary_pairs(ma,mb,system):
    """Yield every affine sign assignment once as two full binary bitsets."""
    particular,kernel=affine_basis(system)
    d=len(ma)
    words=[0,0]
    for s,w in enumerate((ma,mb)):
        for i,value in enumerate(w):
            if value==-2: words[s]|=(1<<i)|(1<<(i+d))
            elif value==0: words[s]|=1<<(i+d)
    def changes(vector):
        masks=[0,0]
        for bit,(s,i) in enumerate(system['variables']):
            if vector>>bit&1: masks[s]^=(1<<i)|(1<<(i+d))
        return masks
    initial=changes(particular)
    words=[w^change for w,change in zip(words,initial)]
    deltas=[changes(v) for v in kernel]
    for index in range(system['affine_assignments']):
        if index:
            changed=(index & -index).bit_length()-1
            words[0]^=deltas[changed][0];words[1]^=deltas[changed][1]
        yield words[0],words[1]


def enumerate_affine_lift(request, seconds):
    """Bounded diagnostic enumeration, not a production search backend.

    Enumerate EVERY solution of the necessary parity system in Gray order,
    checking full periodic correlations exactly with integer bitsets. A
    timeout is UNKNOWN. UNSAT is returned only after the COMPLETE affine
    space is exhausted. SAT must pass the unmodified independent verifier.
    No known PQCP seed, stochastic move, or Z3 query is used in this probe.
    """
    if seconds<=0: raise ValueError('positive budget required')
    start=perf_counter()
    p=TargetContentProfile(**request['target'])
    ma,mb=tuple(request['middle_a']),tuple(request['middle_b'])
    metrics=middle_metrics(ma,mb,p)
    system=orientation_parity_system(ma,mb,metrics['residual_target'])
    record=dict(status='UNKNOWN',verified=False,method='exact_affine_enumeration',
                request=request,metrics=metrics,parity_system=system,enumerated=0,
                budget_seconds=seconds)
    def finish(status):
        record.update(status=status,measured_wall_seconds=perf_counter()-start,
                      censored=status=='UNKNOWN')
        return record
    if not system['consistent']:
        record['reason']='inconsistent necessary GF(2) equations'
        return finish('UNSAT')
    L,d=p.L,p.L//2
    target=target_profile(L,p.k,p.eta)
    wanted=tuple(L-target[u]//2 for u in range(1,d))
    word_mask=(1<<L)-1
    popcount=int.bit_count if hasattr(int,'bit_count') else lambda x: bin(x).count('1')
    for index,(a,b) in enumerate(affine_binary_pairs(ma,mb,system)):
        if index%1024==0 and perf_counter()-start>=seconds:
            record['reason']='enumeration wall budget; unvisited assignments remain'
            return finish('UNKNOWN')
        record['enumerated']+=1
        for u,mismatches in enumerate(wanted,1):
            ra=((a>>u)|(a<<(L-u))) & word_mask
            rb=((b>>u)|(b<<(L-u))) & word_mask
            if popcount(a^ra)+popcount(b^rb)!=mismatches: break
        else:
            A,B=(tuple((word>>i)&1 for i in range(L)) for word in (a,b))
            check=verify_pqcp(A,B)
            assert check.is_valid and check.profile==target
            assert compress_binary_sequence(A,2)==ma and compress_binary_sequence(B,2)==mb
            assert pair_content(A,B)==(p.a_even_ones,p.a_odd_ones,p.b_even_ones,p.b_odd_ones)
            record.update(verified=True,A=A,B=B,profile=check.profile)
            return finish('SAT')
    record['reason']='all affine assignments exhaustively rejected by exact full PACF'
    return finish('UNSAT')


def orientations_with_rows(middle, rows):
    """Count exactly the orientations of one middle word with given row sums.

    L=4n, n odd. Negative middle entries fix a one in both antipodal rows.
    Each zero can allocate its one to either of the two same-parity rows.
    If the even group has z zeros and needs r ones in row 0, its count is
    binomial(z,r), independently of the odd group. No PACF assumption here.
    """
    if not middle or len(middle)%4 != 2 or any(type(x) is not int or x not in (-2,0,2) for x in middle):
        raise ValueError('factor-2 word for L=4n, n odd required')
    n = len(middle)//2
    if len(rows)!=4 or any(type(x) is not int or abs(x)>n or (n-x)%2 for x in rows):
        raise ValueError('invalid row sums')
    fixed, zeros = [0]*4, [0]*2
    for i, value in enumerate(middle):
        if value == -2:
            fixed[i%4] += 1
            fixed[(i+len(middle))%4] += 1
        elif value == 0:
            zeros[i%2] += 1
    degrees = [(n-x)//2 for x in rows]
    count = 1
    for parity in (0,1):
        r, s = degrees[parity]-fixed[parity], degrees[parity+2]-fixed[parity+2]
        if min(r,s)<0 or r+s != zeros[parity]:
            return 0
        count *= comb(zeros[parity],r)
    return count


def middle_metrics(ma, mb, profile):
    """Exact size of the fixed middle lift before remaining PACF equations.

    Distinct labelled row-pairs specify disjoint sets of full binary words,
    hence their binomial counts can be summed without overlap. This count
    does NOT estimate solution probability or imply a hard runtime bound.
    """
    ma, mb = tuple(ma), tuple(mb)
    L, d = profile.L, profile.L//2
    if L%4 or (L//4)%2 != 1 or len(ma)!=d or len(mb)!=d:
        raise ValueError('L=4n, n odd required')
    if any(type(x) is not int or x not in (-2,0,2) for x in ma+mb):
        raise ValueError('invalid middle alphabet')
    target = target_profile(L,profile.k,profile.eta)
    if tuple(a+b for a,b in zip(pacf(ma),pacf(mb))) != fold_profile(target,2):
        raise ValueError('not an exact middle PACF match')
    expected = (profile.a_even_ones,profile.a_odd_ones,
                profile.b_even_ones,profile.b_odd_ones)
    actual = tuple(sum(2 if x==-2 else 1 if x==0 else 0 for x in w[p::2])
                   for w in (ma,mb) for p in (0,1))
    if actual != expected:
        raise ValueError('middle content mismatch')
    n = L//4
    ca,cb = (tuple(w[i]+w[i+n] for i in range(n)) for w in (ma,mb))
    eligible = tuple(pair for pair in compatible_cross_compressions(ca,cb,profile.k,profile.eta,profile)
                     if middle_matches_cross(ma,pair[0]) and middle_matches_cross(mb,pair[1]))
    cross_count = sum(orientations_with_rows(ma,ra)*orientations_with_rows(mb,rb)
                      for ra,rb in set(eligible))
    zeros = ma.count(0)+mb.count(0)
    if zeros != d:
        raise RuntimeError('half-shift invariant failed')
    assert 0 <= cross_count <= 1<<zeros
    return dict(L=L, orientation_bits=zeros, unconstrained_orientations=1<<zeros,
                cross_patterns=len(eligible), cross_constrained_orientations=cross_count,
                independent_residual_shifts=list(range(1,d//2)),
                residual_target=residual_target(L,profile.k,profile.eta),
                cross=eligible)


def _binary_worker(connection, request):
    try:
        from solver.compressed_lift import solve_compressed_lift
        import z3
        connection.send(dict(kind='ready',z3=z3.get_version_string()))
        seconds = connection.recv()
        start = perf_counter()
        p = TargetContentProfile(**request['target'])
        ma, mb = tuple(request['middle_a']), tuple(request['middle_b'])
        metrics = middle_metrics(ma,mb,p)
        connection.send(dict(kind='progress',stats=dict(stage='binary',metrics=metrics)))
        result = solve_compressed_lift(ma,mb,p.k,p.eta,
            max(1,int((seconds-(perf_counter()-start))*1000)),p,cross_compressions=metrics['cross'])
        record = asdict(result)
        if result.status == 'SAT':
            check = verify_pqcp(result.a,result.b)
            assert check.is_valid and check.profile == target_profile(p.L,p.k,p.eta)
            assert compress_binary_sequence(result.a,2)==ma and compress_binary_sequence(result.b,2)==mb
            assert pair_content(result.a,result.b)==(p.a_even_ones,p.a_odd_ones,p.b_even_ones,p.b_odd_ones)
            record.update(A=result.a,B=result.b,profile=check.profile)
        connection.send(dict(kind='final',stats=dict(record,metrics=metrics)))
    except BaseException:
        connection.send(dict(kind='final',stats=dict(status='ERROR',verified=False,reason=traceback.format_exc())))
    finally:
        connection.close()


def probe_binary(request, seconds):
    """Fresh exact Z3 retry, with wall limit also covering model construction."""
    if seconds <= 0:
        raise ValueError('positive budget required')
    ctx=mp.get_context('spawn')
    parent,child=ctx.Pipe()
    process=ctx.Process(target=_binary_worker,args=(child,request))
    launch=perf_counter()
    process.start(); child.close()
    record=dict(status='UNKNOWN',verified=False,stage='startup')
    complete=False
    try:
        if not parent.poll(30):
            raise RuntimeError('worker startup timeout')
        ready=parent.recv()
        if ready['kind']!='ready':
            raise RuntimeError(str(ready))
        startup=perf_counter()-launch
        start=perf_counter()
        parent.send(seconds)
        while perf_counter()-start < seconds:
            if parent.poll(min(.05,max(0,seconds-(perf_counter()-start)))):
                message=parent.recv(); record.update(message['stats'])
                if message['kind']=='final':
                    complete=True
                    break
        record.update(censored=not complete,measured_wall_seconds=perf_counter()-start,
                      startup_seconds=startup,budget_seconds=seconds,request=request)
        if not complete:
            record.update(status='UNKNOWN',verified=False,reason='external wall-clock deadline')
        return record
    finally:
        if process.is_alive(): process.terminate()
        process.join(timeout=5)
        if process.is_alive(): process.kill(); process.join()
        parent.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-manifest',type=Path,required=True)
    parser.add_argument('--seconds-per-child',type=float,default=120)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--enumerate-affine',action='store_true',
                        help='Diagnostic exhaustive GF(2) subspace check instead of Z3')
    args=parser.parse_args()
    if args.seconds_per_child<=0: parser.error('positive budget required')
    from solver.checkpoint import atomic_write_json
    from solver.search_runner import append_verified_solution_if_new
    requests=json.loads(args.probe_manifest.read_text())['requests']
    for request in requests:
        p=TargetContentProfile(**request['target'])
        if p.L!=68: parser.error('this continuation probe is restricted to L68')
        middle_metrics(request['middle_a'],request['middle_b'],p)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    atomic_write_json(args.output_dir/'inputs.json',dict(requests=requests,
        seconds_per_child=args.seconds_per_child,
        policy=('Exhaustive exact affine subspace diagnostic' if args.enumerate_affine else
                'Fresh binary Z3 retries of preselected unresolved middle models; no search parameter changes')))
    records=[]
    with (args.output_dir/'runs.jsonl').open('x') as stream:
        for index,request in enumerate(requests):
            result=(enumerate_affine_lift if args.enumerate_affine else probe_binary)(request,args.seconds_per_child)
            # Save verified A/B BEFORE touching the official result file.
            atomic_write_json(args.output_dir/('child_{:03d}.json'.format(index)),result)
            if result.get('verified'):
                a,b=tuple(result['A']),tuple(result['B'])
                if not verify_pqcp(a,b).is_valid: raise RuntimeError('independent verification failed')
                result['new_solution_written']=append_verified_solution_if_new(68,a,b,Path(__file__).resolve().parents[1])
            stream.write(json.dumps(result)+'\n');stream.flush()
            records.append(result)
            print('child={} k={} eta={} status={} wall={:.3f}s verified={}'.format(
                index,request['target']['k'],request['target']['eta'],result['status'],
                result['measured_wall_seconds'],result.get('verified',False)),flush=True)
            if result['status']=='ERROR': raise RuntimeError(result['reason'])
    atomic_write_json(args.output_dir/'summary.json',dict(
        children=len(records),statuses={s:sum(r['status']==s for r in records) for s in ('SAT','UNSAT','UNKNOWN')},
        verified=sum(bool(r.get('verified')) for r in records),
        new_solutions=sum(bool(r.get('new_solution_written')) for r in records),
        wall_seconds=sum(r['measured_wall_seconds'] for r in records)))


if __name__=='__main__':
    main()
