"""Exact weight filtering and nearest-distance ranking of CRT support families."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from .ternary_support_search import support_signature


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/cyclic_support.c'
    build = root/'build'
    build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('cyclic_support_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix)
        os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_uint64)
    lib.pqcp_nearest_supports.argtypes = [pointer, ctypes.c_int]*3 + [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, pointer, ctypes.c_uint64, pointer]
    lib.pqcp_nearest_supports.restype = ctypes.c_int
    lib.pqcp_support_layer.argtypes = [pointer, ctypes.c_int]*3 + [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
        pointer, ctypes.c_uint64, pointer]
    lib.pqcp_support_layer.restype = ctypes.c_int
    return lib


def nearest_supports(family, reference, weight, *, capacity=8, minimum_distance=0):
    """Enumerate the ENTIRE family, return capped ALL minimum-distance ties.

    Missing farther supports are not an UNSAT result. ``complete_nearest``
    describes only whether every minimum-distance tie fits in the output.
    ``weight_matches`` counts all admissible weights at all distances.
    """
    if (type(family.d) is not int or family.d not in (3, 5, 11, 43)
            or type(family.base) is not int or not 0 <= family.base < 1 << family.d
            or type(family.support_a) is not int or not 0 <= family.support_a < 1 << family.d
            or type(family.k) is not int or not 1 <= family.k <= family.d//2
            or not 1 <= len(family.choices) <= 3
            or any(not group or len(group) > 129 for group in family.choices)
            or type(reference) is not int or not 0 <= reference < 1 << family.d
            or type(weight) is not int or not 0 <= weight <= family.d
            or type(minimum_distance) is not int or not 0 <= minimum_distance <= family.d
            or type(capacity) is not int or not 1 <= capacity <= 4096):
        raise ValueError('supported family, d-bit reference, weight and capacity in 1..4096 required')
    if any(type(m) is not int or not 0 <= m < 1 << family.d
           for group in family.choices for m in group):
        raise ValueError('CRT components must be d-bit integers')
    started = perf_counter()
    array = ctypes.c_uint64
    groups = list(family.choices) + [(0,)]*(3-len(family.choices))
    arrays = [(array*len(group))(*group) for group in groups]
    output = (array*capacity)()
    stats = (array*3)()
    library = _library()
    construction = perf_counter()-started
    started = perf_counter()
    distance = library.pqcp_support_layer(arrays[0], len(groups[0]),
        arrays[1], len(groups[1]), arrays[2], len(groups[2]), family.base,
        reference, weight, minimum_distance, output, capacity, stats)
    elapsed = perf_counter()-started
    if distance < 0 and distance != -2:
        raise RuntimeError('native support enumeration failed')
    masks = tuple(output[:stats[2]])
    signature = support_signature(family.support_a, family.d) ^ (1 << (family.k-1))
    for mask in masks:
        if (bin(mask).count('1') != weight or bin(mask ^ reference).count('1') != distance
                or distance < minimum_distance
                or support_signature(mask, family.d) != signature
                or (bin(mask ^ family.support_a).count('1') & 1) != 1):
            raise RuntimeError('CRT support failed independent integer PACF audit')
    if len(set(masks)) != len(masks):
        raise RuntimeError('duplicate literal support in native output')
    return dict(masks=masks, distance=None if distance == -2 else distance,
        family_size=family.size, weight_matches=int(stats[0]), nearest_ties=int(stats[1]),
        complete_nearest=stats[1] <= capacity, construction_seconds=construction,
        enumeration_seconds=elapsed)


def all_weighted_norm_supports(family,weight):
    """All weight matches of a generic cyclic norm, with literal norm audits."""
    from .sds_norm_completion import cyclic_binary_norm
    if (family.d not in (3,5,11,43) or family.norm_target is None
            or type(weight) is not int or not 0<=weight<=family.d
            or not 1<=len(family.choices)<=3 or any(not g or len(g)>129 for g in family.choices)):
        raise ValueError('generic supported norm family and exact binary support weight required')
    u=ctypes.c_uint64;groups=list(family.choices)+[(0,)]*(3-len(family.choices))
    arrays=[(u*len(g))(*g) for g in groups];output=(u*family.size)();stats=(u*2)()
    function=_library().pqcp_all_weighted_norm_supports
    function.argtypes=[ctypes.POINTER(u),ctypes.c_int]*3+[u,ctypes.c_int,ctypes.c_int,ctypes.POINTER(u),u,ctypes.POINTER(u)]
    function.restype=ctypes.c_int
    status=function(arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
        family.base,family.d,weight,output,family.size,stats)
    if status or stats[0]!=family.size:raise RuntimeError('generic weighted norm family not completely enumerated')
    masks=tuple(map(int,output[:stats[1]]))
    if len(set(masks))!=len(masks) or any(bin(m).count('1')!=weight or cyclic_binary_norm(m,family.d)!=family.norm_target for m in masks):
        raise RuntimeError('weighted generic norm output failed independent literal norm audit')
    return dict(complete=True,masks=masks,visited=int(stats[0]),weight_matches=len(masks))


def dihedral_support_key(mask,d):
    if d not in (3,5,11,43) or type(mask) is not int or not 0<=mask<1<<d:raise ValueError('supported literal support required')
    function=_library().pqcp_dihedral_support_key;function.argtypes=[ctypes.c_uint64,ctypes.c_int];function.restype=ctypes.c_uint64
    value=int(function(mask,d))
    if value>=1<<d:raise RuntimeError('native dihedral support key input audit failed')
    return value


def dihedral_support_keys(masks,d):
    masks=tuple(masks)
    if d not in (3,5,11,43) or any(type(m) is not int or not 0<=m<1<<d for m in masks):
        raise ValueError('supported literal supports required')
    if not masks:return ()
    u=ctypes.c_uint64;n=len(masks);output=(u*n)()
    function=_library().pqcp_dihedral_support_keys
    function.argtypes=[ctypes.POINTER(u),u,ctypes.c_int,ctypes.POINTER(u)];function.restype=ctypes.c_int
    if function((u*n)(*masks),n,d,output):raise RuntimeError('native bulk dihedral key audit failed')
    return tuple(map(int,output))


def signed_complement_supports(family,profiles,negative):
    """Every norm/weight support surviving certified signed edge-cut bounds."""
    return _signed_supports(family,profiles,negative,single=False)


def signed_norm_supports(family,profile,negative,*,native_audit=False):
    """Same certified cuts for ONE signed norm, without constraining the complement."""
    profile=tuple(profile);d=family.d
    return _signed_supports(family,(profile,(d-profile[0],)+(0,)*(d-1)),(negative,0),single=True,native_audit=native_audit)


def _signed_supports(family,profiles,negative,*,single,native_audit=False):
    from .sds_norm_completion import cyclic_binary_norm
    profiles=tuple(tuple(p) for p in profiles);negative=tuple(negative);d=family.d
    if (d not in (3,5,11,43) or family.norm_target is None or len(profiles)!=2 or len(negative)!=2
            or any(len(p)!=d or any(type(v) is not int or not -d<=v<=d for v in p) for p in profiles)
            or any(p[u]!=p[-u] for p in profiles for u in range(1,d))
            or sum(p[0] for p in profiles)!=d or any(not 0<=p[0]<=d for p in profiles)
            or any(type(n) is not int or not 0<=n<=p[0] for n,p in zip(negative,profiles))):
        raise ValueError('two complementary-support signed norm profiles and negative counts required')
    u=ctypes.c_uint64;i=ctypes.c_int;groups=list(family.choices)+[(0,)]*(3-len(family.choices))
    arrays=[(u*len(g))(*g) for g in groups];outputs=(u*family.size)();proofs=(u*(2*family.size))();stats=(u*4)()
    function=_library().pqcp_signed_complement_supports
    function.argtypes=[ctypes.POINTER(u),ctypes.c_int]*3+[u,i,i,ctypes.POINTER(i),ctypes.POINTER(i),i,i,ctypes.POINTER(u),ctypes.POINTER(u),u,ctypes.POINTER(u)]
    function.restype=i
    status=function(arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
        family.base,d,profiles[0][0],(i*d)(*profiles[0]),None if single else (i*d)(*profiles[1]),*negative,outputs,proofs,family.size,stats)
    if status or stats[0]!=family.size or stats[1]!=stats[2]+stats[3]:raise RuntimeError('signed support filter did not cover its entire norm family')
    certificates=[];all_bits=(1<<d)-1
    for at in range(stats[3]):
        mask,code=int(proofs[2*at]),int(proofs[2*at+1]);shift=code>>8;side=(code>>4)&1;reason=code&15
        if single and side:raise RuntimeError('single-norm filter attempted to constrain an unknown complement norm')
        support=mask if side==0 else all_bits^mask
        intersection=bin(support&(((support<<shift)|(support>>(d-shift)))&all_bits)).count('1')
        wanted=profiles[side][shift];minority=min(negative[side],profiles[side][0]-negative[side])
        valid=(abs(wanted)>intersection if reason==0 else
               intersection-wanted>4*minority if reason==1 else
               (intersection-wanted)%2!=0 if reason==2 else False)
        if not valid:raise RuntimeError('native signed support exclusion failed independent Python cut-bound audit')
        certificates.append(dict(mask=mask,side=side,shift=shift,reason=reason,
            unsigned_intersection=intersection,wanted_signed_intersection=wanted,minority_count=minority))
    masks=tuple(map(int,outputs[:stats[2]]))
    if type(native_audit) is not bool:raise ValueError('boolean native full-norm audit switch required')
    invalid=(not audit_literal_support_norms(masks,d,family.norm_target,profiles[0][0])) if native_audit else any(
        bin(m).count('1')!=profiles[0][0] or cyclic_binary_norm(m,d)!=family.norm_target for m in masks)
    if len(set(masks))!=len(masks) or invalid:
        raise RuntimeError('retained signed-support mask failed full literal norm/weight audit')
    return dict(complete=True,masks=masks,visited=int(stats[0]),weight_matches=int(stats[1]),
                exclusions=tuple(certificates),excluded_supports=len(certificates))


def audit_literal_support_norms(masks,d,target,weight):
    """Every output audited exactly, by independent full GF2 convolution."""
    masks=tuple(masks)
    if (d not in (3,5,11,43) or type(target) is not int or not 0<=target<1<<d
            or type(weight) is not int or not 0<=weight<=d
            or any(type(m) is not int or not 0<=m<1<<d for m in masks)):
        raise ValueError('supported literal supports, target and exact weight required')
    if not masks:return True
    u=ctypes.c_uint64;n=len(masks);bad=u()
    fn=_library().pqcp_audit_literal_support_norms
    fn.argtypes=[ctypes.POINTER(u),u,ctypes.c_int,u,ctypes.c_int,ctypes.POINTER(u)];fn.restype=ctypes.c_int
    status=fn((u*n)(*masks),n,d,target,weight,ctypes.byref(bad))
    if status not in (0,1):raise RuntimeError('native literal support norm audit rejected input')
    return status==0
