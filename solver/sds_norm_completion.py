"""Complete fixed-side supplementary difference sets from exact field norms.

U U*+V V* has diagonal (r+s) mod 2 and every other entry lambda mod 2.
CRT enumerates ALL necessary V; exact integer intersections then decide SDS.
This creates new D-optimal-43 ingredients, never certifies a full L86 PQCP.
"""
import ctypes
from collections import defaultdict
from time import perf_counter

from .cyclic_support import cyclic_norm_family,rotation_representatives
from .cyclic_support_native import _library


def sds_ingredient_key(mask,d):
    if type(d) is not int or d not in (3,5,11,43) or type(mask) is not int or not 0<=mask<1<<d:
        raise ValueError('supported prime length and literal support mask required')
    function=_library().pqcp_sds_ingredient_key
    function.argtypes=[ctypes.c_uint64,ctypes.c_int];function.restype=ctypes.c_uint64
    value=int(function(mask,d))
    if value>=1<<d:raise RuntimeError('SDS ingredient quotient returned invalid key')
    return value


def native_odd_cross_vectors(even,odd,d=43):
    if type(d) is not int or not 3<=d<=43 or d%2!=1 or any(type(m) is not int or not 0<=m<1<<d for m in (even,odd)):
        raise ValueError('odd length and literal row masks required')
    function=_library().pqcp_odd_cross_vectors
    function.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    function.restype=ctypes.c_int
    width=d//2+1;output=(ctypes.c_int*(d*width))()
    if function(even,odd,d,output)!=0:raise RuntimeError('native odd correlation generation failed')
    groups=defaultdict(list)
    for delta in range(d):groups[tuple(output[delta*width:(delta+1)*width])].append(delta)
    return dict(groups)


def cyclic_binary_norm(mask,d):
    if type(d) is not int or d<3 or d%2!=1 or type(mask) is not int or not 0<=mask<1<<d:
        raise ValueError('odd length and literal binary support mask required')
    all_bits=(1<<d)-1
    result=bin(mask).count('1')%2
    for u in range(1,d//2+1):
        rotated=((mask<<u)|(mask>>(d-u)))&all_bits
        if bin(mask&rotated).count('1')%2:result|=(1<<u)|(1<<(d-u))
    return result


def complete_sds_support(fixed,d,other_weight,lambda_value,*,rotation_quotient=True):
    if (type(d) is not int or d not in (3,5,11,43) or type(fixed) is not int or not 0<=fixed<1<<d
            or type(other_weight) is not int or not 0<=other_weight<=d
            or type(lambda_value) is not int or not 0<=lambda_value<=2*d
            or type(rotation_quotient) is not bool):
        raise ValueError('supported odd prime length, supports, weights and nonnegative lambda required')
    started=perf_counter();fixed_weight=bin(fixed).count('1')
    if fixed_weight*(fixed_weight-1)+other_weight*(other_weight-1)!=lambda_value*(d-1):
        return dict(complete=True,masks=(),literal_family_size=0,family_size=0,visited=0,
                    weight_matches=0,correlation_checks=0,exact_models=0,
                    rotation_quotient=rotation_quotient,elapsed=perf_counter()-started,
                    reason='exact sum of intersection counts contradicts constant lambda')
    desired=((fixed_weight+other_weight)%2)|((((1<<d)-1)^1)*(lambda_value%2))
    norm_target=desired^cyclic_binary_norm(fixed,d)
    family=cyclic_norm_family(norm_target,d);literal_size=family.size
    if rotation_quotient:family=rotation_representatives(family)
    u=ctypes.c_uint64
    groups=list(family.choices)+[(0,)]*(3-len(family.choices))
    arrays=[(u*len(group))(*group) for group in groups]
    function=_library().pqcp_complete_sds
    function.argtypes=[ctypes.POINTER(u),ctypes.c_int]*3+[u,u,ctypes.c_int,ctypes.c_int,
        ctypes.c_int,ctypes.POINTER(u),u,ctypes.POINTER(u)]
    function.restype=ctypes.c_int
    # Allocate the whole certified finite family. This preserves all literal
    # outputs even in unexpectedly dense small cases, without a truncation cap.
    output=(u*family.size)();stats=(u*5)()
    status=function(arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
                    family.base,fixed,d,other_weight,lambda_value,output,family.size,stats)
    if status!=0 or stats[0]!=family.size or stats[3]!=stats[4]:
        raise RuntimeError('SDS completion did not cover the entire certified family')
    masks=tuple(int(v) for v in output[:stats[4]])
    if len(set(masks))!=len(masks):raise RuntimeError('SDS CRT output has duplicate literal preimages')
    all_bits=(1<<d)-1
    for mask in masks:
        if bin(mask).count('1')!=other_weight or cyclic_binary_norm(mask,d)!=norm_target:
            raise RuntimeError('SDS output failed exact weight/norm audit')
        for shift in range(1,d):
            def rotate(m):return ((m<<shift)|(m>>(d-shift)))&all_bits
            if bin(fixed&rotate(fixed)).count('1')+bin(mask&rotate(mask)).count('1')!=lambda_value:
                raise RuntimeError('SDS output failed independent ALL-shift integer intersection audit')
    return dict(complete=True,masks=masks,literal_family_size=literal_size,family_size=family.size,
                visited=int(stats[0]),weight_matches=int(stats[1]),correlation_checks=int(stats[2]),
                exact_models=int(stats[3]),rotation_quotient=rotation_quotient,
                norm_target=norm_target,elapsed=perf_counter()-started,
                scope='ALL fixed-side constant-intersection SDS completions, '
                      'modulo independent V rotation only when requested; not all U or L86 PQCPs.')


def intersection_profile(mask,d):
    if type(d) is not int or d<3 or d%2!=1 or type(mask) is not int or not 0<=mask<1<<d:
        raise ValueError('odd length and literal support mask required')
    all_bits=(1<<d)-1
    return (bin(mask).count('1'),)+tuple(bin(mask&(((mask<<u)|(mask>>(d-u)))&all_bits)).count('1')
                                        for u in range(1,d))


def complete_binary_intersection_profile(wanted,*,rotation_quotient=True,certified_psd=True,obstruction_only=False,
                                         prime_norm=False,prime_limit=50000):
    """ALL binary supports with a given integer intersection profile.

    Rotation affects no PACF coordinate. The full preimage set is retained
    when rotation_quotient=False. Empty finite fibers are not global exclusions.
    """
    wanted=tuple(wanted);d=len(wanted)
    if (d not in (3,5,11,43) or any(type(v) is not int for v in wanted)
            or not 0<=wanted[0]<=d or any(wanted[u]!=wanted[-u] for u in range(1,d))
            or type(rotation_quotient) is not bool or type(certified_psd) is not bool or type(obstruction_only) is not bool
            or type(prime_norm) is not bool or type(prime_limit) is not int or not 3<=prime_limit<=65000):
        raise ValueError('supported prime length and symmetric integral intersection profile required')
    started=perf_counter();weight=wanted[0]
    common=dict(masks=(),complete=True,literal_family_size=0,family_size=0,visited=0,weight_matches=0,
                correlation_checks=0,exact_models=0,rotation_quotient=rotation_quotient,wanted=wanted)
    lower=max(0,2*weight-d);upper=weight if weight in (0,d) else weight-1
    if any(not lower<=v<=upper for v in wanted[1:]):
        return dict(common,elapsed=perf_counter()-started,reason='exact prime-length intersection range obstruction')
    if sum(wanted[1:])!=weight*(weight-1):
        return dict(common,elapsed=perf_counter()-started,reason='exact intersection sum obstruction')
    if certified_psd:
        from .certified_profile_psd import profile_psd_obstruction
        obstruction=profile_psd_obstruction(wanted)
        if obstruction is not None:
            return dict(common,elapsed=perf_counter()-started,reason='certified negative PSD upper bound',
                        psd_certificate=obstruction)
    if prime_norm:
        from .prime_norm_obstruction import prime_norm_obstruction
        obstruction=prime_norm_obstruction(wanted,limit=prime_limit)
        if obstruction is not None:
            return dict(common,elapsed=perf_counter()-started,reason='single integer norm fails p-squared Hensel divisibility',
                        prime_norm_certificate=obstruction)
    if obstruction_only:
        return dict(common,complete=False,status='NECESSARY_FILTER_PASSED',elapsed=perf_counter()-started,
                    reason='necessary bounds passed; no preimage enumeration performed')
    norm_target=sum((v%2)<<i for i,v in enumerate(wanted))
    family=cyclic_norm_family(norm_target,d);literal_size=family.size
    if rotation_quotient:family=rotation_representatives(family)
    u=ctypes.c_uint64;groups=list(family.choices)+[(0,)]*(3-len(family.choices))
    arrays=[(u*len(g))(*g) for g in groups]
    target=(ctypes.c_int*(d//2))(*wanted[1:d//2+1]);output=(u*family.size)();stats=(u*5)()
    function=_library().pqcp_complete_intersection_profile
    function.argtypes=[ctypes.POINTER(u),ctypes.c_int]*3+[u,ctypes.c_int,ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),ctypes.POINTER(u),u,ctypes.POINTER(u)]
    function.restype=ctypes.c_int
    code=function(arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
                  family.base,d,weight,target,output,family.size,stats)
    if code!=0 or stats[0]!=family.size or stats[3]!=stats[4]:
        raise RuntimeError('intersection-profile completion did not exhaust certified norm family')
    masks=tuple(int(v) for v in output[:stats[4]])
    if len(set(masks))!=len(masks) or any(intersection_profile(m,d)!=wanted for m in masks):
        raise RuntimeError('intersection-profile output failed independent ALL-shift integer audit')
    return dict(common,masks=masks,literal_family_size=literal_size,family_size=family.size,
                norm_target=norm_target,visited=int(stats[0]),weight_matches=int(stats[1]),
                correlation_checks=int(stats[2]),exact_models=int(stats[3]),elapsed=perf_counter()-started,
                scope='ALL binary preimages of this exact intersection profile, '
                      'modulo rotation only if requested; not all profiles or PQCPs.')


def complete_sds_many_supports(fixed,d,other_weight,lambda_value,*,rotation_quotient=True):
    """Fix up to three SDS blocks and completely decide the remaining block."""
    fixed=tuple(fixed)
    if (type(d) is not int or d not in (3,5,11,43) or not 1<=len(fixed)<=3
            or any(type(m) is not int or not 0<=m<1<<d for m in fixed)
            or type(other_weight) is not int or not 0<=other_weight<=d
            or type(lambda_value) is not int or not 0<=lambda_value<=4*d):
        raise ValueError('one to three literal fixed blocks and supported integer SDS parameters required')
    profiles=tuple(intersection_profile(m,d) for m in fixed)
    wanted=(other_weight,)+tuple(lambda_value-sum(p[u] for p in profiles) for u in range(1,d))
    result=complete_binary_intersection_profile(wanted,rotation_quotient=rotation_quotient)
    return dict(result,fixed=fixed,lambda_value=lambda_value,
                scope='ALL remaining SDS block completions of these fixed blocks; '
                      'modulo independent remaining-block rotation only if requested. No global exclusion.')
