"""Complete bounded changes in one half and arbitrary changes in another.

All 12 ordered (varied, free) roles preserve the UNION under full PQCP
equivalence. Hamming balls include arbitrary positions and weight changes.
Every fixed triple is completed exactly; no homometric or phase preimage cap.
"""
from itertools import combinations
from math import comb
from time import perf_counter

from .correlation import normalize_binary_sequence
from .half_row_completion import complete_three_parity_rows,_target_index
from .verifier import verify_pqcp


def iter_half_flip_masks(d,radius):
    for count in range(radius+1):
        for positions in combinations(range(d),count):yield sum(1<<i for i in positions)


def two_half_row_fibers(a,b,*,radius=2):
    """Generate every distinct eligible literal fixed-triple fiber.

    Content-excluded triples are UNSAT by the full DC/ALT necessary index.
    The stream retains all literal triples, modulo only identical repeats.
    """
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b));length=len(a);d=length//2
    if len(b)!=length or length!=2*d or d not in (3,5,11,43) or type(radius) is not int or not 0<=radius<=min(4,d):
        raise ValueError('supported equal twice-prime rows and radius 0..min(4,d) required')
    if verify_pqcp(a,b).is_valid:raise ValueError('two-half centers must be non-solutions')
    halves=(a[::2],a[1::2],b[::2],b[1::2]);masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in halves)
    weights=tuple(map(sum,halves));zero_free_seen=set();ordinal=0
    for varied in range(4):
        for free in range(4):
            if free==varied:continue
            target_index=_target_index(length,free)
            for flips in iter_half_flip_masks(d,radius):
                changed=masks[varied]^flips;counts=list(weights);counts[varied]=bin(changed).count('1')
                fixed_counts=tuple(v for i,v in enumerate(counts) if i!=free)
                ordinal+=1
                if fixed_counts not in target_index:
                    yield dict(kind='CONTENT_EXCLUDED',varied=varied,free=free,flips=flips,ordinal=ordinal);continue
                key=(free,)+tuple(changed if i==varied else masks[i] for i in range(4) if i!=free)
                # Nonzero flips in distinct varied roles alter distinct known
                # rows, so their literal triples cannot coincide. XOR also
                # makes distinct flip masks injective within one varied role.
                # Only zero-flip baselines repeat across the three roles.
                if not flips and free in zero_free_seen:
                    yield dict(kind='IDENTICAL_TRIPLE',varied=varied,free=free,flips=flips,ordinal=ordinal);continue
                if not flips:zero_free_seen.add(free)
                trial=list(halves);trial[free]=None
                trial[varied]=tuple((changed>>i)&1 for i in range(d))
                yield dict(kind='ELIGIBLE',varied=varied,free=free,flips=flips,halves=tuple(trial),
                           fixed_masks=key[1:],ordinal=ordinal)


def complete_two_parity_halves(a,b,*,radius=2,certified_psd=True,trace_backend=False,joint_gram=False,finite_trace=False,batched_psd=False):
    if (type(certified_psd) is not bool or type(trace_backend) is not bool or type(joint_gram) is not bool
            or type(finite_trace) is not bool or type(batched_psd) is not bool):
        raise ValueError('boolean completion backend switches required')
    started=perf_counter();counts=dict(content_excluded=0,identical_triples=0,completed_fibers=0,
                                       norm_visited=0,psd_certificates=0)
    witnesses=[];seen=set();total=0
    for job in two_half_row_fibers(a,b,radius=radius):
        total+=1
        if job['kind']=='CONTENT_EXCLUDED':counts['content_excluded']+=1;continue
        if job['kind']=='IDENTICAL_TRIPLE':counts['identical_triples']+=1;continue
        out=complete_three_parity_rows(job['halves'],free_component=job['free'],
                                       certified_psd=certified_psd,trace_backend=trace_backend,joint_gram=joint_gram,
                                       finite_trace=finite_trace,batched_psd=batched_psd)
        if not out['complete']:raise RuntimeError('fixed-triple fiber incomplete')
        counts['completed_fibers']+=1;counts['norm_visited']+=out['norm_visited']
        counts['psd_certificates']+=sum('psd_certificate' in r['result'] for r in out['profile_records'])
        for hit in out['witnesses']:
            if hit['words'] in seen:continue
            seen.add(hit['words']);witnesses.append(dict(hit,varied_component=job['varied'],varied_flips=job['flips']))
    d=len(a)//2;expected=12*sum(comb(d,r) for r in range(radius+1))
    if total!=expected:raise RuntimeError('ordered-role Hamming ball enumeration incomplete')
    return dict(complete=True,status='SAT' if witnesses else 'UNSAT_DECLARED_TWO_HALF_REGION',
                witnesses=witnesses,ordered_roles=12,radius=radius,generated_triples=total,**counts,
                raw_assignments_with_overlaps=total*(1<<d),elapsed=perf_counter()-started,
                scope='UNION of all 12 varied/free roles: varied half has arbitrary-position flips up to radius; '
                      'free half entirely arbitrary; other two literal halves fixed. All necessary contents and targets. '
                      'No global exclusion or new-discovery-speed claim.')
