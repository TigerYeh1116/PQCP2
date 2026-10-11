"""Native Boolean elimination with the same replayable proofs as the auditor.

The default audits every proof using independent Python symbolic-set products.
audit_certificates=False measures the native algorithm without that diagnostic
overhead; this experimental backend is not enabled in production.
"""
import ctypes
from functools import lru_cache
import hashlib
from math import comb
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .mod8_crossbred import boolean_equations, replay_certificate
from .mod8_model_reference import _validated_coefficients
from .mod8_model_reference import pack_projection_native


@lru_cache(maxsize=1)
def library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/mod8_crossbred.c'
    build = root/'build'; build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path = build/('mod8_crossbred_'+digest+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix); os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                            '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                            str(source), '-o', temporary], check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    result = ctypes.CDLL(str(path))
    u = ctypes.c_uint32; p = ctypes.POINTER(u); integer = ctypes.c_int
    wide = ctypes.POINTER(ctypes.c_uint64)
    result.pqcp_crossbred_tail3.argtypes = [u, p, p, integer, integer, ctypes.POINTER(integer),
                                          integer, integer, p, wide, wide, ctypes.POINTER(integer)]
    result.pqcp_crossbred_tail3.restype = integer
    result.pqcp_crossbred_lookup.argtypes = [p, wide, integer, integer, integer, integer,
                                            integer, p, p, ctypes.POINTER(integer), ctypes.POINTER(integer)]
    result.pqcp_crossbred_lookup.restype = integer
    return result


def _native_basis(coefficients, tail, multiplier_axes):
    constant, linear, quadratic, q = _validated_coefficients(coefficients)
    if type(tail) is not int or not 0 <= tail <= min(q, 8):
        raise ValueError('tail between zero and min(q,8) required')
    axes = tuple(range(q)) if multiplier_axes is None else tuple(multiplier_axes)
    if len(set(axes)) != len(axes) or any(type(j) is not int or not 0 <= j < q for j in axes):
        raise ValueError('distinct literal multiplier axes required')
    width = max((constant, *linear, *quadratic)).bit_length()
    columns = sum(comb(q, degree) for degree in range(min(q, 3)+1))
    words = (columns+63)//64
    proof_words = max(1, (width*(q+1)+63)//64)
    capacity = max(1, width*(len(axes)+1))
    u = ctypes.c_uint32; wide = ctypes.c_uint64; integer = ctypes.c_int
    monomials = (u*columns)(); rows = (wide*(capacity*words))()
    proofs = (wide*(capacity*proof_words))(); meta = (integer*6)()
    code = library().pqcp_crossbred_tail3(constant, (u*q)(*linear), (u*(q*q))(*quadratic),
        q, tail, (integer*len(axes))(*axes), len(axes), capacity, monomials, rows, proofs, meta)
    if code:
        raise RuntimeError('native Crossbred construction failed; no exclusions performed')
    if tuple(meta[1:2]) != (columns,) or meta[3] != words or meta[4] != proof_words:
        raise RuntimeError('native Crossbred metadata mismatch')
    return dict(q=q,tail=tail,width=width,axes=axes,columns=columns,words=words,
                proof_words=proof_words,monomials=monomials,rows=rows,proofs=proofs,meta=meta)


def native_tail_consequences(coefficients, tail=6, *, multiplier_axes=None, audit_certificates=True):
    if type(audit_certificates) is not bool:
        raise ValueError('boolean certificate audit option required')
    raw=_native_basis(coefficients,tail,multiplier_axes)
    q=raw['q'];width=raw['width'];axes=raw['axes'];columns=raw['columns'];words=raw['words']
    proof_words=raw['proof_words'];monomials=raw['monomials'];rows=raw['rows'];proofs=raw['proofs'];meta=raw['meta']
    originals = boolean_equations(coefficients) if audit_certificates else None
    count = int(meta[0]); prefix = q-tail; prefix_mask = (1 << prefix)-1
    consequences = []; packed = {}
    for equation in range(count):
        terms = []
        for word in range(words):
            value = int(rows[equation*words+word])
            while value:
                low = value & -value; value ^= low
                terms.append(int(monomials[word*64+low.bit_length()-1]))
        proof = sum(int(proofs[equation*proof_words+j]) << (64*j) for j in range(proof_words))
        if audit_certificates and replay_certificate(originals, q, proof) != frozenset(terms):
            raise RuntimeError('native consequence failed independent symbolic certificate replay')
        if any(bin(term >> prefix).count('1') > 1 for term in terms):
            raise RuntimeError('native consequence still contains nonlinear tail terms')
        consequences.append(dict(terms=tuple(terms), proof=hex(proof)))
        for term in terms:
            suffix = term >> prefix
            block = suffix.bit_length() if suffix else 0
            key = term & prefix_mask
            packed[key] = packed.get(key, 0) ^ (1 << (block*count+equation))
    return dict(q=q, tail=tail, prefix=prefix, input_equations=width,
                multiplied_rows=int(meta[5]), monomials=columns, multiplier_axes=axes,
                eliminated_columns=columns-int(meta[2]), consequence_count=count,
                consequences=tuple(consequences), packed=tuple(packed.items()),
                all_certificates_independently_replayed=audit_certificates)


def lookup_tail_consequences(coefficients, tail=6, *, prefix_multipliers=8, audit_certificates=False):
    """Native lookup buffers for <=32 consequences independent of old tests.

    Dropping unselected equations weakens a necessary condition. The original
    mod8 candidate check remains required after this additional prefix filter.
    Buffers use uint32 bit fields, including bit 31, without numeric conversion.
    """
    q=len(coefficients[1])
    if (type(tail) is not int or not 0<=tail<=min(q,8) or type(prefix_multipliers) is not int
            or not 0<=prefix_multipliers<=24 or type(audit_certificates) is not bool):
        raise ValueError('bounded tail and prefix multiplier counts required')
    prefix=q-tail;axes=tuple(range(min(prefix_multipliers,prefix)))+tuple(range(prefix,q))
    raw=_native_basis(coefficients,tail,axes)
    _,projected,_=pack_projection_native(coefficients,max(1,tail))
    # For tail zero, the packing helper must use the actual zero-dimensional
    # quotient, not a one-variable tail. Tail-Q spans coincide for sizes 0/1.
    low=prefix//2;high=prefix-low;L=1<<low;H=1<<high
    stride=L+H+low*H+high*L
    u=ctypes.c_uint32;integer=ctypes.c_int
    table=(u*((tail+1)*stride))();selected=(integer*32)();count=integer()
    code=library().pqcp_crossbred_lookup(raw['monomials'],raw['rows'],raw['meta'][0],raw['columns'],
        raw['words'],q,tail,(u*601)(*projected),table,selected,ctypes.byref(count))
    if code:raise RuntimeError('native Crossbred lookup packing failed; no exclusions performed')
    proofs=[];terms=[];originals=boolean_equations(coefficients) if audit_certificates else None
    for number in selected[:count.value]:
        proof=sum(int(raw['proofs'][number*raw['proof_words']+j])<<(64*j) for j in range(raw['proof_words']))
        proofs.append(hex(proof))
        if audit_certificates:
            values=[]
            for word in range(raw['words']):
                value=int(raw['rows'][number*raw['words']+word])
                while value:
                    bit=value&-value;value^=bit
                    values.append(int(raw['monomials'][word*64+bit.bit_length()-1]))
            if (any((v>>prefix)&((v>>prefix)-1) for v in values)
                    or replay_certificate(originals,q,proof)!=frozenset(values)):
                raise RuntimeError('selected lookup equation failed independent proof replay')
            terms.append(tuple(values))
    return dict(q=q,tail=tail,prefix=prefix,low=low,L=L,H=H,stride=stride,table=table,
                selected_indices=tuple(selected[:count.value]),selected_count=count.value,
                proofs=tuple(proofs),terms=tuple(terms) if audit_certificates else None,
                all_certificates_independently_replayed=audit_certificates)


def specialize_lookup(lookup,bits):
    """Independent table-layout evaluator, useful for complete CPU auditing."""
    if type(bits) is not int or not 0<=bits<1<<lookup['prefix']:
        raise ValueError('literal prefix assignment required')
    low=lookup['low'];L=lookup['L'];H=lookup['H'];lo=bits&(L-1);hi=bits>>low
    stride=lookup['stride'];table=lookup['table'];values=[]
    for block in range(lookup['tail']+1):
        offset=block*stride;value=int(table[offset+lo])^int(table[offset+L+hi])
        remaining=lo
        while remaining:
            bit=remaining&-remaining;remaining^=bit
            value^=int(table[offset+L+H+(bit.bit_length()-1)*H+hi])
        remaining=hi
        while remaining:
            bit=remaining&-remaining;remaining^=bit
            value^=int(table[offset+L+H+low*H+(bit.bit_length()-1)*L+lo])
        values.append(value)
    return values[0],tuple(values[1:])
