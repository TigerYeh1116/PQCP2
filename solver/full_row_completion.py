"""Fix a complete binary A row; B is entirely free (both parity halves).

CRT Z_(2d)=Z_2 x Z_d makes binary rows two complementary trit rows.
Their unary integer norms decouple. For each support and its complement,
ANY independently verified pair of signed norm witnesses gives a full B.
First-witness SAT is not enumeration of every novel full equivalence class.
UNSAT requires exhaustive support coverage and no capped UNKNOWN branch.
"""
from functools import lru_cache
from math import isqrt
from time import perf_counter

from .correlation import normalize_binary_sequence
from .cyclic_support import cyclic_norm_family,rotation_representatives
from .cyclic_support_native import all_weighted_norm_supports,signed_complement_supports
from .structured_energy import integer_periodic_autocorrelation_profile as trit_profile
from .joint_gram_pruning import _fourier_interval
from .prime_norm_obstruction import prime_norm_obstruction
from .ternary_system_native import prepare_native_system,run_native_system
from .verifier import verify_pqcp


def unary_sign_completion(support,d,wanted,negative,budget,*,two_adic=False,cut_projection=False):
    """Audited one-row plan, optionally completed by lifted zero-field norms."""
    plan=prepare_native_system((support,0),d,wanted,(negative,0))
    # Do cheap affine contradictions first. This cost threshold schedules
    # a necessary test; no branch is excluded merely for being inexpensive.
    if cut_projection and plan['consistent'] and plan['estimated_assignments']>=4096:
        from .support_cut_projection import support_cut_obstruction
        proof=support_cut_obstruction((support,),d,wanted,(negative,))
        if proof is not None:return dict(status='UNSAT_FIXED_SUPPORT',complete=True,words=None,assignments=0,cut_certificate=proof)
    if two_adic and plan['consistent'] and plan['meta'][2]>=10:
        from .two_adic_norm_completion import prefer_lifted_unary,lifted_unary_signs
        if prefer_lifted_unary(plan,wanted):
            # Preserve inexpensive early witnesses before constructing a
            # lifted root search. UNKNOWN from this probe is not an exclusion.
            probe=run_native_system(plan,min(128,budget),quadratic=False,conditioned_hybrid=False,hybrid_tail=0)
            if probe['status']!='UNKNOWN' or budget<=probe['assignments']:return probe
            result=lifted_unary_signs(plan,wanted,negative,budget-probe['assignments'])
            if result is not None:
                result['assignments']+=probe['assignments'];result['early_probe_assignments']=probe['assignments']
            if result is not None:return result
    return run_native_system(plan,budget)


def crt_trit_rows(word):
    word=tuple(normalize_binary_sequence(word));d=len(word)//2
    if len(word)!=2*d or d not in (3,5,11,43):raise ValueError('supported twice-prime binary row required')
    signs=tuple(1-2*v for v in word)
    c=tuple((signs[2*j]+signs[(2*j+d)%(2*d)])//2 for j in range(d))
    g=tuple((signs[2*j]-signs[(2*j+d)%(2*d)])//2 for j in range(d))
    return c,g


def reconstruct_crt_binary(c,g):
    c,g=tuple(c),tuple(g);d=len(c)
    if (d not in (3,5,11,43) or len(g)!=d or any(type(v) is not int or v not in (-1,0,1) for v in c+g)
            or any((u!=0)+(v!=0)!=1 for u,v in zip(c,g))):
        raise ValueError('complementary signed trit rows required')
    bits=[None]*(2*d)
    for j,(u,v) in enumerate(zip(c,g)):
        bits[2*j]=(1-u-v)//2;bits[(2*j+d)%(2*d)]=(1-u+v)//2
    return tuple(bits)


@lru_cache(maxsize=32)
def _prepared_full_row(a):
    c,g=crt_trit_rows(a)
    return c,g,tuple(trit_profile(c)),tuple(trit_profile(g))


def full_row_target_profiles(a,k,eta):
    return _target_from_prepared(_prepared_full_row(tuple(normalize_binary_sequence(a))),k,eta)


def _target_from_prepared(prepared,k,eta):
    c,g,nc,ng=prepared;d=len(c)
    if type(k) is not int or not 1<=k<d or type(eta) is not int or eta not in (-1,1):raise ValueError('strict target required')
    tau=k*((d+1)//2)%d;wc=d-nc[0];wg=d-ng[0]
    wanted_c=(wc,)+tuple(eta*int(u in (tau,-tau%d))-nc[u] for u in range(1,d))
    wanted_g=(wg,)+tuple(eta*((-1)**k)*int(u in (tau,-tau%d))-ng[u] for u in range(1,d))
    values=(d+2*eta-sum(c)**2,d+2*eta*((-1)**k)-sum(g)**2)
    contents=[]
    for weight,value in zip((wc,wg),values):
        if value<0 or isqrt(value)**2!=value:return None
        sigma=isqrt(value)
        if sigma>weight or (weight-sigma)%2:return None
        contents.append((weight-sigma)//2)
    if sum(wanted_c)!=values[0] or sum(wanted_g)!=values[1]:raise RuntimeError('CRT unary target scalar audit failed')
    return dict(wanted_c=wanted_c,wanted_g=wanted_g,negative=tuple(contents),
                positive_scalar_sums=tuple(isqrt(v) for v in values),k=k,eta=eta)


def signed_norm_psd_obstruction(profile):
    profile=tuple(profile);d=len(profile)
    if (d not in (3,5,11,43) or any(type(v) is not int or not -d<=v<=d for v in profile)
            or any(profile[u]!=profile[-u] for u in range(1,d))):raise ValueError('supported signed integral symmetric norm required')
    for frequency in range(1,d//2+1):
        lower,upper=_fourier_interval(profile,frequency)
        if upper<0:return dict(profile=profile,frequency=frequency,upper_bound_numerator=upper,
                               reason='SIGNED_NORM_NEGATIVE_OUTWARD_PSD',denominator=1<<40)
    return None


def complete_fixed_binary_row_target(a,*,k,eta,certified_psd=True,prime_norm=True,max_assignments=2**31,on_support=None,support_bounds=False,two_adic=False,cut_projection=False):
    """One full B witness, or certified exhaustion/explicit UNKNOWN.

    Both positive scalar sums are a proven B sign/half-period-shift gauge.
    A remains literally fixed. Support rotations are B even shifts, so all
    existence cases are retained. No signed-preimage cutoff proves UNSAT.
    """
    if (type(certified_psd) is not bool or type(prime_norm) is not bool or type(support_bounds) is not bool or type(two_adic) is not bool or type(cut_projection) is not bool
            or type(max_assignments) is not int or not 1<=max_assignments<=2**31
            or (on_support is not None and not callable(on_support))):
        raise ValueError('boolean necessary-condition switches and assignment budget 1..2**31 required')
    a=tuple(normalize_binary_sequence(a));d=len(a)//2;spec=_target_from_prepared(_prepared_full_row(a),k,eta);tick=perf_counter()
    out=dict(k=k,eta=eta,complete=True,words=None,support_orbits=0,sign_fibers=0,assignments=0,
             unknown_sign_fibers=0,profile=spec,status='UNSAT_FIXED_FULL_A',
             two_adic=two_adic,
             cut_projection=cut_projection,
             scope='All 2d binary bits of B, A literally fixed, one strict k/eta; first witness only when SAT.')
    if spec is None:return dict(out,reason='exact DC/ALT/midpoint content obstruction',elapsed=perf_counter()-tick)
    for profile in (spec['wanted_c'],spec['wanted_g']):
        if any(abs(v)>profile[0] for v in profile[1:]):
            return dict(out,reason='single-trit integer intersection magnitude obstruction',elapsed=perf_counter()-tick)
        if certified_psd:
            proof=signed_norm_psd_obstruction(profile)
            if proof is not None:return dict(out,reason=proof['reason'],psd_certificate=proof,elapsed=perf_counter()-tick)
        if prime_norm:
            proof=prime_norm_obstruction(profile)
            if proof is not None:return dict(out,reason='single-trit p-squared Hensel obstruction',prime_certificate=proof,elapsed=perf_counter()-tick)
        if two_adic:
            from .two_adic_norm_completion import two_adic_norm_obstruction
            proof=two_adic_norm_obstruction(profile)
            if proof is not None:return dict(out,reason='single-trit mod-four norm obstruction',two_adic_certificate=proof,elapsed=perf_counter()-tick)
    desired=sum((v%2)<<u for u,v in enumerate(spec['wanted_c']))
    desired_g=sum((v%2)<<u for u,v in enumerate(spec['wanted_g']))
    if desired^desired_g!=(1<<d)-1:raise RuntimeError('complementary support norm parity identity failed')
    family=rotation_representatives(cyclic_norm_family(desired,d))
    supports=(signed_complement_supports(family,(spec['wanted_c'],spec['wanted_g']),spec['negative']) if support_bounds else
              all_weighted_norm_supports(family,spec['wanted_c'][0]));all_bits=(1<<d)-1
    out.update(norm_family_size=family.size,norm_visited=supports['visited'],eligible_support_orbits=len(supports['masks']),
               raw_weight_support_orbits=supports['weight_matches'],support_exclusions=supports.get('exclusions',()),
               support_bounds=support_bounds)
    for mask in supports['masks']:
        out['support_orbits']+=1;parts=[];undecided=False;attempts=[]
        # The sign problems are independent; no relative-phase matching is
        # left after c/g complementary supports and unary norms are fixed.
        for support,wanted,negative in ((mask,spec['wanted_c'],spec['negative'][0]),
                                       (all_bits^mask,spec['wanted_g'],spec['negative'][1])):
            result=unary_sign_completion(support,d,wanted,negative,max_assignments,two_adic=two_adic,cut_projection=cut_projection)
            out['sign_fibers']+=1;out['assignments']+=result['assignments']
            attempts.append(dict(support=support,wanted=wanted,negative=negative,result=result))
            if result['status']=='UNSAT_FIXED_SUPPORT':break
            if result['status']=='UNKNOWN':
                out['unknown_sign_fibers']+=1;undecided=True;break
            row=tuple(result['words'][0])
            if (tuple(trit_profile(row))!=wanted or row.count(-1)!=negative
                    or sum((v!=0)<<i for i,v in enumerate(row))!=support):raise RuntimeError('unary signed norm witness failed full integer audit')
            parts.append(row)
        else:
            b=reconstruct_crt_binary(*parts);check=verify_pqcp(a,b)
            if not check.is_valid or check.nonzero_shifts!=(k,2*d-k) or check.profile[k]!=4*eta:
                raise RuntimeError('CRT binary full-row completion failed original full verifier')
            if on_support is not None:on_support(dict(mask=mask,attempts=attempts,full_words=(a,b)))
            return dict(out,status='SAT_FULL_PQCP',complete=False,words=(a,b),support_mask=mask,
                        elapsed=perf_counter()-tick,reason='First fully verified binary B; remaining support orbits not enumerated.')
        if on_support is not None:on_support(dict(mask=mask,attempts=attempts,complete=not undecided))
        if undecided:out['complete']=False
    if out['unknown_sign_fibers']:out['status']='UNKNOWN_FIXED_FULL_A'
    return dict(out,elapsed=perf_counter()-tick)
