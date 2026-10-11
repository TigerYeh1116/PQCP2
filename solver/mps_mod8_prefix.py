"""Integer-only full mod8 candidate enumeration on the GPU.

Only q<=24 COMPLETE raw affine spaces are supported. Every prefix and
every projected tail nullspace member is visited. Overflow is DEFERRED,
never UNSAT. Returned indices are audited against the original polynomials.
This experimental module is not enabled in the production sampler.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .ternary_system_native import _library as native_library


def quadratic_coefficients(plan):
    q=int(plan['meta'][2]);u=ctypes.c_uint32
    if not plan['consistent'] or plan['selected']>=0 or not 0<=q<=24:
        raise ValueError('consistent raw affine plan through dimension 24 required')
    constant=u();linear=(u*q)();quadratic=(u*(q*q))()
    status=native_library().pqcp_quadratic_plan(plan['mask_array'],plan['base'],plan['toggles'],q,
        plan['d'],plan['wanted'],plan['negative'],ctypes.byref(constant),linear,quadratic)
    if status:raise RuntimeError('native quadratic construction failed')
    return int(constant.value),tuple(map(int,linear)),tuple(map(int,quadratic))


def evaluate(coefficients,index):
    constant,linear,quadratic=coefficients;q=len(linear);value=constant
    for i in range(q):
        if index>>i&1:
            value^=linear[i]
            for j in range(i+1,q):
                if index>>j&1:value^=quadratic[i*q+j]
    return value


def packed_coefficients(coefficients,tail):
    constant,linear,quadratic=coefficients;q=len(linear);tail=min(tail,q);basis={}
    for i in range(q-tail,q):
        for j in range(i+1,q):
            value=quadratic[i*q+j]
            while value:
                bit=value.bit_length()-1
                if bit not in basis:basis[bit]=value;break
                value^=basis[bit]
    def reduce(value):
        for bit in sorted(basis,reverse=True):
            if value>>bit&1:value^=basis[bit]
        return value
    raw=[constant]+list(linear)+[0]*(24-q)+[0]*576
    for i in range(q):
        for j in range(q):raw[25+24*i+j]=quadratic[i*q+j]
    return raw,[reduce(v) for v in raw],tail


@lru_cache(maxsize=1)
def shader():
    import torch
    source=Path(__file__).resolve().parents[1]/'csrc/mod8_prefix.metal'
    return torch.mps.compile_shader(source.read_text())


def enumerate_mod8_candidates(plans,*,tail=6,capacity=256,native_packing=True,crossbred=False,prefix_multipliers=8,preparation_workers=1,bit_sliced=False):
    import torch
    plans=tuple(plans)
    if (not plans or len(plans)>1024 or type(tail) is not int or not 1<=tail<=8
            or type(capacity) is not int or not 1<=capacity<=65536 or type(native_packing) is not bool
            or type(crossbred) is not bool or type(prefix_multipliers) is not int or not 0<=prefix_multipliers<=24
            or type(preparation_workers) is not int or not 1<=preparation_workers<=16 or type(bit_sliced) is not bool
            or (crossbred and bit_sliced)):
        raise ValueError('1..1024 raw plans, tail 1..8, bounded capacity required')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS device unavailable; no GPU exclusions performed')
    if native_packing:
        from .mod8_model_reference import pack_projection_native,library as pack_library
        pack_library()
        pack=pack_projection_native
    else:pack=packed_coefficients
    if crossbred:
        from .mod8_crossbred_native import lookup_tail_consequences,library as crossbred_library
        crossbred_library()
    def prepare(plan):
        c=quadratic_coefficients(plan)
        if bit_sliced:
            from .mod8_model_reference import _pack_bit_coefficients_from_native
            t=min(tail,len(c[1]));equations=plan['d']//2+2
            r,truth,chunks=_pack_bit_coefficients_from_native(c,t,equations);p=None
            lookup=dict(table=truth,equations=equations,chunks=chunks)
        else:
            r,p,t=pack(c,tail)
            lookup=lookup_tail_consequences(c,t,prefix_multipliers=prefix_multipliers) if crossbred else None
        return c,r,p,t,lookup
    if preparation_workers==1:
        prepared=[prepare(p) for p in plans]
    else:
        # ctypes releases the GIL inside the native kernels; all output
        # buffers are private. Only the parent thread submits MPS work.
        with ThreadPoolExecutor(max_workers=preparation_workers) as pool:
            prepared=list(pool.map(prepare,plans))
    coefficients=[item[0] for item in prepared]
    raw=[];projected=[];metadata=[];maximum_prefix=1
    for i,(c,r,p,t,_) in enumerate(prepared):
        q=len(c[1]);raw.append(r);projected.append(p)
        metadata.append((q,t,capacity,i*capacity));maximum_prefix=max(maximum_prefix,1<<(q-t))
    device=torch.device('mps')
    if not bit_sliced:
        raw_tensor=torch.tensor(raw,dtype=torch.int32,device=device)
        projected_tensor=torch.tensor(projected,dtype=torch.int32,device=device)
        meta_tensor=torch.tensor(metadata,dtype=torch.int32,device=device)
    output=torch.empty((len(plans),capacity),dtype=torch.int32,device=device)
    counts=torch.zeros(len(plans),dtype=torch.int32,device=device)
    if bit_sliced:
        packet=(ctypes.c_uint32*(872*len(plans)))()
        for i,(_,raw,_,_,value) in enumerate(prepared):
            header=(ctypes.c_uint32*7)(*metadata[i],value['equations'],value['chunks'],0)
            ctypes.memmove(ctypes.addressof(packet)+4*872*i,ctypes.addressof(header),4*7)
            ctypes.memmove(ctypes.addressof(packet)+4*(872*i+7),ctypes.addressof(raw),4*601)
            ctypes.memmove(ctypes.addressof(packet)+4*(872*i+608),ctypes.addressof(value['table']),4*len(value['table']))
        packet_tensor=torch.frombuffer(packet,dtype=torch.int32).to(device)
        use_specialized=all(len(c[1])==21 and t==6 and value['equations']==23 for c,_,_,t,value in prepared)
        kernel=shader().pqcp_mod8_bit_tail_q21_t6 if use_specialized else shader().pqcp_mod8_bit_tail_prefix
        kernel(packet_tensor,output,counts,
            threads=(maximum_prefix,len(plans)),group_size=(64,1))
    elif crossbred:
        lookups=[item[4] for item in prepared]
        total=sum(len(value['table']) for value in lookups)
        if total>=2**29:raise ValueError('Crossbred lookup buffer exceeds bounded GPU allocation')
        combined=(ctypes.c_uint32*total)();lookup_meta=[];offset=0
        for value in lookups:
            count=len(value['table']);ctypes.memmove(ctypes.addressof(combined)+4*offset,ctypes.addressof(value['table']),4*count)
            lookup_meta.append((offset,value['low'],value['L'],value['H'],value['stride'],value['selected_count']))
            offset+=count
        lookup_tensor=torch.frombuffer(combined,dtype=torch.int32).to(device)
        lookup_meta_tensor=torch.tensor(lookup_meta,dtype=torch.int32,device=device)
        shader().pqcp_mod8_crossbred_prefix(raw_tensor,projected_tensor,meta_tensor,output,counts,lookup_tensor,lookup_meta_tensor,
            threads=(maximum_prefix,len(plans)),group_size=(64,1))
    else:
        shader().pqcp_mod8_prefix(raw_tensor,projected_tensor,meta_tensor,output,counts,
            threads=(maximum_prefix,len(plans)),group_size=(64,1))
    torch.mps.synchronize();numbers=counts.cpu().tolist();results=[]
    # Copy only the complete retained model lists. Unused capacity and
    # overflowed partial lists are neither evidence nor useful host data.
    maximum_complete=max((n for n in numbers if 0<=n<=capacity),default=0)
    values=output[:,:maximum_complete].contiguous().cpu().tolist() if maximum_complete else [() for _ in plans]
    for c,number,items in zip(coefficients,numbers,values):
        if number>capacity:
            results.append(dict(complete=False,status='DEFERRED_OVERFLOW',candidate_count=number,indices=None));continue
        indices=tuple(sorted(items[:number]));q=len(c[1])
        if len(set(indices))!=number or any(not 0<=j<1<<q or evaluate(c,j)!=0 for j in indices):
            raise RuntimeError('GPU candidates failed independent original quadratic audit')
        results.append(dict(complete=True,status='MOD8_COMPLETE',candidate_count=number,indices=indices))
    return tuple(results)


def complete_mod8_models(plan,candidates):
    """Exact integer completion only after a COMPLETE GPU candidate list."""
    from .structured_energy import integer_periodic_autocorrelation_profile as pacf
    if not candidates['complete']:
        return dict(status='UNKNOWN',words=None,reason='GPU model buffer overflow; CPU fallback required')
    q=int(plan['meta'][2]);d=plan['d'];checks=0
    for index in candidates['indices']:
        signs=[int(v) for v in plan['base']]
        for j in range(q):
            if index>>j&1:
                for s in range(2):signs[s]^=int(plan['toggles'][2*j+s])
        if tuple(bin(v).count('1') for v in signs)!=tuple(plan['negative']):continue
        words=tuple(tuple((-1 if signs[s]>>i&1 else 1) if plan['masks'][s]>>i&1 else 0 for i in range(d)) for s in range(2))
        checks+=1
        if tuple(x+y for x,y in zip(pacf(words[0]),pacf(words[1])))==tuple(plan['wanted']):
            return dict(status='SAT_COMPRESSED',words=words,integer_models_checked=checks)
    return dict(status='UNSAT_FIXED_SUPPORT',words=None,integer_models_checked=checks,
                scope='Complete raw affine space through all original mod8 constraints and all retained integer models.')
