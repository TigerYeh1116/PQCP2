"""CPU full-zero-set auditor for experimental GPU prefix enumeration."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


@lru_cache(maxsize=1)
def library():
    root=Path(__file__).resolve().parents[1];source=root/'csrc/mod8_model_reference.c'
    build=root/'build';build.mkdir(exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path=build/('mod8_model_reference_'+digest+('.dylib' if sys.platform=='darwin' else '.so'))
    if not path.exists():
        fd,temp=tempfile.mkstemp(dir=build,suffix=path.suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',temp],check=True,capture_output=True)
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    result=ctypes.CDLL(str(path));u=ctypes.c_uint32
    result.pqcp_mod8_reference.argtypes=[u,ctypes.POINTER(u),ctypes.POINTER(u),ctypes.c_int,
        ctypes.c_uint64,ctypes.POINTER(u),ctypes.POINTER(ctypes.c_uint64)]
    result.pqcp_mod8_reference.restype=ctypes.c_int
    result.pqcp_pack_mod8_projection.argtypes=[u,ctypes.POINTER(u),ctypes.POINTER(u),ctypes.c_int,ctypes.c_int,
                                              ctypes.POINTER(u),ctypes.POINTER(u)]
    result.pqcp_pack_mod8_projection.restype=ctypes.c_int
    result.pqcp_pack_mod8_bit_tail.argtypes=[ctypes.POINTER(u),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(u)]
    result.pqcp_pack_mod8_bit_tail.restype=ctypes.c_int
    result.pqcp_pack_mod8_bit_coefficients.argtypes=[u,ctypes.POINTER(u),ctypes.POINTER(u),ctypes.c_int,
        ctypes.c_int,ctypes.c_int,ctypes.POINTER(u),ctypes.POINTER(u)]
    result.pqcp_pack_mod8_bit_coefficients.restype=ctypes.c_int
    return result


def _validated_coefficients(coefficients):
    constant,linear,quadratic=coefficients;q=len(linear)
    if (not 0<=q<=24 or len(quadratic)!=q*q
            or any(type(v) is not int or not 0<=v<2**32 for v in (constant,*linear,*quadratic))
            or any(quadratic[i*q+i] for i in range(q))
            or any(quadratic[i*q+j]!=quadratic[j*q+i] for i in range(q) for j in range(i+1,q))):
        raise ValueError('symmetric uint32 Boolean quadratic system through dimension 24 required')
    return constant,linear,quadratic,q


def all_mod8_indices(coefficients,capacity=65536):
    constant,linear,quadratic,q=_validated_coefficients(coefficients)
    if type(capacity) is not int or not 1<=capacity<=2**24:
        raise ValueError('bounded candidate capacity required')
    u=ctypes.c_uint32;output=(u*capacity)();count=ctypes.c_uint64()
    code=library().pqcp_mod8_reference(constant,(u*q)(*linear),(u*(q*q))(*quadratic),q,capacity,output,ctypes.byref(count))
    if code:raise RuntimeError('full CPU modular reference failed')
    return dict(complete=count.value<=capacity,candidate_count=count.value,
                indices=tuple(sorted(output[:count.value])) if count.value<=capacity else None)


def pack_projection_native(coefficients,tail):
    constant,linear,quadratic,q=_validated_coefficients(coefficients)
    if type(tail) is not int or not 1<=tail<=8:
        raise ValueError('tail between one and eight required')
    tail=min(q,tail)
    u=ctypes.c_uint32;raw=(u*601)();projected=(u*601)()
    code=library().pqcp_pack_mod8_projection(constant,(u*q)(*linear),(u*(q*q))(*quadratic),q,tail,raw,projected)
    if code:raise RuntimeError('native modular projection packing failed')
    return list(raw),list(projected),tail


def pack_bit_tail_native(raw,q,tail,equations):
    if (len(raw)!=601 or type(q) is not int or not 0<=q<=24 or type(tail) is not int
            or not 0<=tail<=min(q,8) or type(equations) is not int or not 1<=equations<=25
            or any(type(v) is not int or not 0<=v<1<<equations for v in raw)):
        raise ValueError('bounded exact packed syndrome and tail required')
    u=ctypes.c_uint32;chunks=((1<<tail)+31)//32;table=(u*((equations+tail)*chunks))()
    if library().pqcp_pack_mod8_bit_tail((u*601)(*raw),q,tail,equations,table):
        raise RuntimeError('native exact tail truth construction failed')
    return table,chunks


def _pack_bit_coefficients_from_native(coefficients,tail,equations):
    """Internal path: coefficients produced by the checked native plan API.

    The C kernel additionally checks every coefficient against the physical
    syndrome width. Public arbitrary packed arrays use pack_bit_tail_native.
    """
    constant,linear,quadratic=coefficients;q=len(linear)
    if (not 0<=q<=24 or len(quadratic)!=q*q or type(tail) is not int or not 0<=tail<=min(q,8)
            or type(equations) is not int or not 1<=equations<=25):
        raise ValueError('bounded native quadratic coefficient dimensions required')
    u=ctypes.c_uint32;chunks=((1<<tail)+31)//32;raw=(u*601)();table=(u*((equations+tail)*chunks))()
    code=library().pqcp_pack_mod8_bit_coefficients(constant,(u*q)(*linear),(u*(q*q))(*quadratic),
        q,tail,equations,raw,table)
    if code:raise RuntimeError('native bit-tail coefficient packing failed; no exclusions performed')
    return raw,table,chunks
