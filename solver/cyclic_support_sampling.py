"""Conditional index sampling inside an exact weight-filtered CRT family.

This chooses a literal norm preimage instead of a nearest-distance winner.
It is search scheduling, never a global UNSAT or uniformity claim over all
support pairs. Independent integer parity/PACF checks audit every sample.
"""
import ctypes
from functools import lru_cache
from .cyclic_support import NormSupportFamily,rotation_representatives

from .cyclic_support_native import _library,nearest_supports
from .ternary_support_search import support_signature

DEFAULT_PHASE_RANK_PRUNING=True  # Complete rotated fibres; matched pipeline gain 31.84%.


@lru_cache(maxsize=1024)
def _native_group(d,group):
    if not group or len(group)>129 or any(type(mask) is not int or not 0<=mask<1<<d for mask in group):
        raise ValueError('CRT components must be bounded d-bit integer groups')
    return (ctypes.c_uint64*len(group))(*group)


@lru_cache(maxsize=1024)
def _rotation_step(d,group):
    """Certify one rotation's phase translation; iteration proves all shifts."""
    all_bits=(1<<d)-1
    def rotate(value):
        return ((value<<1)|(value>>(d-1)))&all_bits
    indices={mask:j for j,mask in enumerate(group)}
    if len(indices)!=len(group) or rotate(group[0]) not in indices:
        raise RuntimeError('norm phases are not a complete rotation orbit product')
    step=indices[rotate(group[0])]
    if any(rotate(mask)!=group[(j+step)%len(group)] for j,mask in enumerate(group)):
        raise RuntimeError('cyclic phase translation failed its literal certificate')
    return step


def _audit_selected(family,weight,mask):
    if (mask>>family.d or bin(mask).count('1')!=weight
            or support_signature(mask,family.d)!=support_signature(family.support_a,family.d) ^ (1<<(family.k-1))
            or (bin(mask ^ family.support_a).count('1')&1)!=1):
        raise RuntimeError('sampled CRT preimage failed exact independent audit')
    return mask


def weighted_support_at(family,weight,index):
    # The existing API validates the complete family representation and
    # obtains an exact count from its entire literal phase product.
    if family.norm_target is not None:
        raise ValueError('PQCP fixed-side norm family required')
    counted=nearest_supports(family,0,weight,capacity=1)
    count=counted['weight_matches']
    if type(index) is not int or not 0<=index<count:
        raise ValueError('index must select an existing literal weight match')
    return _select(family,weight,index),count


def _select(family,weight,index):
    u=ctypes.c_uint64
    groups=list(family.choices)+[(0,)]*(3-len(family.choices))
    arrays=[(u*len(g))(*g) for g in groups]
    output=u()
    function=_library().pqcp_weighted_support_at
    function.argtypes=[ctypes.POINTER(u),ctypes.c_int]*3+[u,ctypes.c_int,u,ctypes.POINTER(u)]
    function.restype=ctypes.c_int
    status=function(arrays[0],len(groups[0]),arrays[1],len(groups[1]),arrays[2],len(groups[2]),
        family.base,weight,index,ctypes.byref(output))
    mask=int(output.value)
    if status!=1:
        raise RuntimeError('sampled CRT preimage failed exact independent audit')
    return _audit_selected(family,weight,mask)


def _sample_phase_grid(family,weight,rng):
    if (type(family.d) is not int or family.d not in (3,5,11,43)
            or type(family.base) is not int or not 0<=family.base<1<<family.d
            or type(family.support_a) is not int or not 0<=family.support_a<1<<family.d
            or type(family.k) is not int or not 1<=family.k<=family.d//2
            or type(weight) is not int or not 0<=weight<=family.d
            or not 1<=len(family.choices)<=3):
        raise ValueError('supported norm family and exact weight required')
    quotient=rotation_representatives(family)
    first=next((j for j,(a,b) in enumerate(zip(family.choices,quotient.choices)) if len(a)!=len(b)),None)
    groups=tuple(quotient.choices)+((0,),)*(3-len(quotient.choices))
    sizes=tuple(len(g) for g in groups)
    u=ctypes.c_uint64;up=ctypes.POINTER(u);integer=ctypes.c_int
    capacity=sizes[0]*sizes[1]*((sizes[2]+63)//64)
    grid=(u*capacity)();count=u()
    fn=_library().pqcp_weighted_support_grid
    fn.argtypes=[up,integer]*3+[u,integer,up,u,up];fn.restype=integer
    status=fn(_native_group(family.d,groups[0]),sizes[0],_native_group(family.d,groups[1]),sizes[1],
              _native_group(family.d,groups[2]),sizes[2],family.base,weight,grid,capacity,ctypes.byref(count))
    if status:raise RuntimeError('complete rotated weight grid failed')
    blocks=1 if first is None else len(family.choices[first])//len(quotient.choices[first])
    if blocks not in (1,family.d):raise RuntimeError('unexpected norm-phase block count')
    per_block=int(count.value);total=blocks*per_block
    metadata=dict(phase_count_pruning=True,phase_blocks=blocks,phase_block_weight_matches=per_block,
                  counted_family_size=quotient.size)
    if not total:return dict(mask=None,weight_matches=0,family_size=family.size,
        status='UNSAT_WEIGHT_FOR_FIXED_SIDE',complete_count=True,**metadata)
    index=rng.randrange(total);block,rank=divmod(index,per_block)
    rotation=0;shifts=[0,0,0]
    if first is not None:
        width=len(quotient.choices[first]);step=_rotation_step(family.d,family.choices[first])
        if step%width:raise RuntimeError('rotation does not preserve literal phase blocks')
        rotation=block*pow(step//width,-1,family.d)%family.d
        for j in range(first+1,len(family.choices)):
            shifts[j]=_rotation_step(family.d,family.choices[j])*rotation%len(family.choices[j])
    ordinal=u();select=_library().pqcp_weighted_grid_at
    select.argtypes=[up,integer,integer,integer,integer,integer,integer,u,up];select.restype=integer
    if select(grid,*sizes,*shifts,rank,ctypes.byref(ordinal))!=1:
        raise RuntimeError('original literal weighted rank was lost')
    a,remainder=divmod(ordinal.value,sizes[1]*sizes[2]);b,c=divmod(remainder,sizes[2])
    mask=family.base^groups[0][a]^groups[1][b]^groups[2][c]
    if rotation:mask=((mask<<rotation)|(mask>>(family.d-rotation)))&((1<<family.d)-1)
    return dict(mask=_audit_selected(family,weight,mask),weight_matches=total,family_size=family.size,
        selected_index=index,status='SAMPLED_LITERAL_SUPPORT',complete_count=True,phase_block=block,**metadata)


def _phase_count(family,weight):
    """Count ONE orbit transversal, derive all literal counts exactly.

    The first nonzero norm-kernel group has consecutive equal-weight blocks
    of size (2^(f/2)+1)/d. Rotation permutes all other complete groups, so
    every block has the same weight count. An original literal rank I can
    therefore be selected in its block without changing the sampled mask.
    """
    quotient=rotation_representatives(family)
    first=next((j for j,(a,b) in enumerate(zip(family.choices,quotient.choices)) if len(a)!=len(b)),None)
    counted=nearest_supports(quotient,0,weight,capacity=1)
    blocks=1 if first is None else len(family.choices[first])//len(quotient.choices[first])
    if blocks not in (1,family.d):raise RuntimeError('unexpected norm-phase block count')
    return quotient,first,blocks,int(counted['weight_matches'])


def sample_weighted_support(family,weight,rng,*,phase_count_pruning=False,phase_rank_pruning=None):
    if family.norm_target is not None:
        raise ValueError('PQCP fixed-side norm family required')
    if type(phase_count_pruning) is not bool:raise ValueError('boolean exact phase-count pruning flag required')
    if phase_rank_pruning is None:phase_rank_pruning=DEFAULT_PHASE_RANK_PRUNING and phase_count_pruning
    if type(phase_rank_pruning) is not bool:raise ValueError('boolean phase-rank pruning flag required')
    if phase_rank_pruning and not phase_count_pruning:raise ValueError('phase-rank pruning requires phase counting')
    if phase_rank_pruning:return _sample_phase_grid(family,weight,rng)
    if phase_count_pruning:
        quotient,first,blocks,per_block=_phase_count(family,weight)
        count=blocks*per_block
        metadata=dict(phase_count_pruning=True,phase_blocks=blocks,phase_block_weight_matches=per_block,
                      counted_family_size=quotient.size)
        if not count:return dict(mask=None,weight_matches=0,family_size=family.size,
            status='UNSAT_WEIGHT_FOR_FIXED_SIDE',complete_count=True,**metadata)
        index=rng.randrange(count);block,rank=divmod(index,per_block);groups=list(family.choices)
        if first is not None:
            width=len(quotient.choices[first]);groups[first]=groups[first][block*width:(block+1)*width]
        selected=NormSupportFamily(family.d,family.support_a,family.k,family.base,tuple(groups),family.norm_target)
        return dict(mask=_select(selected,weight,rank),weight_matches=count,family_size=family.size,
            selected_index=index,status='SAMPLED_LITERAL_SUPPORT',complete_count=True,phase_block=block,**metadata)
    counted=nearest_supports(family,0,weight,capacity=1)
    count=counted['weight_matches']
    if not count:
        return dict(mask=None,weight_matches=0,family_size=family.size,
                    status='UNSAT_WEIGHT_FOR_FIXED_SIDE',complete_count=True)
    index=rng.randrange(count)
    return dict(mask=_select(family,weight,index),weight_matches=count,family_size=family.size,
                selected_index=index,status='SAMPLED_LITERAL_SUPPORT',complete_count=True)
