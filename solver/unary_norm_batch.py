"""Complete low-dimensional unary sign fibers, with an independent C audit.

All deferred high-dimensional supports remain available to the original
kernel/norm lifts. Packed records retain every input/outcome for replay.
"""
import base64
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
from time import perf_counter

from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .cyclic_support_native import dihedral_support_key,dihedral_support_keys


class CompleteUnarySupportOrbitUnsatCache:
    """In-process complete SINGLE C_B exclusions, never G-only/UNKNOWN.

    Rotation/reflection of a signed row preserves every integer PACF and
    its negative count. Each stored record comes from the independent
    native exhaustion auditor; same-batch unproved guesses are not reused.
    """
    def __init__(self):
        self.context = None
        self._proofs = {}
        self._full_results = {}

    def bind(self, context):
        if self.context is None:
            self.context = context
        elif self.context != context:
            raise ValueError('support orbit proof belongs to a different single norm/count')

    def lookup(self, mask, d):
        return self._proofs.get(dihedral_support_key(mask, d))

    def remember(self, record, d, key=None):
        # Quotient records must refer to a genuine original native proof.
        if record[1]&255 or (record[1]>>8)&255 not in (1, 2, 3, 4, 5):
            return
        if len(self._proofs) < 65536:
            self._proofs.setdefault(dihedral_support_key(record[0], d) if key is None else key, tuple(record))

    def remember_full_exclusion(self, mask, result, d):
        if (result.get('status')!='UNSAT_FIXED_SUPPORT' or not result.get('complete')
                or result.get('words') is not None):
            raise ValueError('complete SINGLE C_B support exclusion required; never SAT/UNKNOWN/G-only')
        key=dihedral_support_key(mask,d)
        if len(self._proofs)<65536 and key not in self._proofs:
            self._proofs[key]=(mask,7<<8,0,result['assignments'],0)
            self._full_results[mask]=dict(result)


@lru_cache(maxsize=1)
def _library():
    root=Path(__file__).resolve().parents[1];sources=[root/'csrc'/name for name in
        ('unary_norm_batch.c','ternary_system.c','ternary_signs.c')]
    digest=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in sources)).hexdigest()[:16]
    build=root/'build';build.mkdir(exist_ok=True);suffix='.dylib' if sys.platform=='darwin' else '.so'
    path=build/('unary_norm_batch_'+digest+suffix)
    if not path.exists():
        fd,temporary=tempfile.mkstemp(dir=build,suffix=suffix);os.close(fd)
        try:
            subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-fPIC',
                '-dynamiclib' if sys.platform=='darwin' else '-shared',*(str(p) for p in sources),'-o',temporary],
                check=True,capture_output=True,timeout=30)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    lib=ctypes.CDLL(str(path));u=ctypes.c_uint64;p=ctypes.POINTER(u);i=ctypes.c_int
    lib.pqcp_unary_norm_batch.argtypes=[p,u,i,ctypes.POINTER(i),i,i,p,p,p]
    lib.pqcp_unary_norm_batch.restype=i
    lib.pqcp_unary_norm_batch_certified.argtypes=[p,u,i,ctypes.POINTER(i),i,i,i,p,p,p]
    lib.pqcp_unary_norm_batch_certified.restype=i
    lib.pqcp_verify_unary_affine_certificate.argtypes=[u,i,ctypes.POINTER(i),i,u,p,i,u]
    lib.pqcp_verify_unary_affine_certificate.restype=i
    return lib


def complete_unary_norm_batch(masks,d,wanted,negative,budget,*,free_limit=12,dihedral_unsat_cache=None,affine_audit=False,linear_certificate=False):
    masks=tuple(masks);wanted=tuple(wanted)
    if (type(d) is not int or not 3<=d<=43 or not d%2 or len(wanted)!=d
            or any(type(v) is not int or not -d<=v<=d for v in wanted)
            or any(wanted[u]!=wanted[-u] for u in range(1,d))
            or any(type(m) is not int or not 0<=m<1<<d or bin(m).count('1')!=wanted[0] for m in masks)
            or len(set(masks))!=len(masks) or type(negative) is not int or not 0<=negative<=wanted[0]
            or type(budget) is not int or not 1<=budget<=2**31 or type(free_limit) is not int or not 0<=free_limit<=12
            or type(affine_audit) is not bool or type(linear_certificate) is not bool or (affine_audit and linear_certificate)):
        raise ValueError('unique supported literal masks, symmetric single norm/count and bounded exact enumeration required')
    if dihedral_unsat_cache is not None and not isinstance(dihedral_unsat_cache,CompleteUnarySupportOrbitUnsatCache):
        raise ValueError('in-process independently audited single norm orbit cache required')
    tick=perf_counter();proofs={};native_masks=[];native_positions=[];keys=()
    if dihedral_unsat_cache is not None:
        dihedral_unsat_cache.bind((d,wanted,negative))
    if dihedral_unsat_cache is None:native_masks=masks
    else:
        keys=dihedral_support_keys(masks,d)
        for index,mask in enumerate(masks):
            proof=dihedral_unsat_cache._proofs.get(keys[index])
            if proof is not None:proofs[index]=proof
            else:native_masks.append(mask);native_positions.append(index)
        native_masks=tuple(native_masks)
    q_limit=min(free_limit,budget.bit_length()-1);u=ctypes.c_uint64;i=ctypes.c_int;n=len(masks)
    m=len(native_masks);records=(u*(5*m))();retained=(u*m)();stats=(u*8)()
    if m:
        status=_library().pqcp_unary_norm_batch_certified((u*m)(*native_masks),m,d,(i*d)(*wanted),negative,q_limit,2 if linear_certificate else int(affine_audit),records,retained,stats)
        if status:raise RuntimeError('batched unary norm classification/independent audit failed: '+str(status))
    if stats[0]!=m or sum(stats[1:4])!=m or stats[4]!=stats[2]+stats[3]:
        raise RuntimeError('unary norm batch did not account for every literal support')
    candidates=[]
    for index in retained[:stats[4]]:
        mask,flags,minus,seen,audited=map(int,records[5*index:5*index+5]);status=flags&255
        if mask!=native_masks[index] or status not in (1,2):raise RuntimeError('batch candidate index/status mismatch')
        word=None
        if status==1:
            word=tuple(0 if not mask>>j&1 else -1 if minus>>j&1 else 1 for j in range(d))
            if minus&~mask or word.count(-1)!=negative or tuple(pacf(word))!=wanted:
                raise RuntimeError('batch SAT carrier failed independent full Python norm/count audit')
        candidates.append((mask,word,int(seen)))
    receipt_blob=bytearray();full_receipts={};blob=bytes(records)
    if dihedral_unsat_cache is not None:
        for position,record in zip(native_positions,struct.iter_unpack('=QQQQQ',blob)):
            dihedral_unsat_cache.remember(record,d,keys[position])
        for index,proof in proofs.items():
            receipt_blob.extend(struct.pack('=QQQQQQ',index,masks[index],proof[0],proof[1],proof[3],proof[4]))
            if proof[0] in dihedral_unsat_cache._full_results:
                full_receipts[str(proof[0])]=dihedral_unsat_cache._full_results[proof[0]]
    return dict(candidates=tuple(candidates),complete_input_partition=True,input_supports=n,
        native_input_supports=m,dihedral_unsat_supports=len(proofs),
        affine_space_equality_audit=affine_audit,
        affine_constraint_certificate=linear_certificate,
        dihedral_unsat_receipts_base64=base64.b64encode(receipt_blob).decode('ascii'),
        dihedral_unsat_receipt_columns=('logical_index','support','source_support','source_flags','source_assignments','source_auditor_assignments'),
        dihedral_unsat_full_source_results=full_receipts,
        unsat_supports=int(stats[1])+len(proofs),sat_supports=int(stats[2]),deferred_supports=int(stats[3]),
        raw_candidates_inspected=int(stats[5]),auditor_candidates_inspected=int(stats[6]),
        free_limit=q_limit,elapsed=perf_counter()-tick,record_sha256=hashlib.sha256(blob).hexdigest(),
        certificate_base64=base64.b64encode(blob).decode('ascii'),certificate_byteorder=sys.byteorder,
        certificate_rows=m,certificate_scope='Native inputs; omitted literal inputs covered by separate complete single-norm dihedral receipts.',
        certificate_uint64_columns=('support','status_reason_rank_free','negative_witness','raw_candidates','auditor_candidates'),
        scope=('Every enumerated C_B-UNSAT has independently checked necessary equations, rank, affine base and independent spanning directions; exact literal enumeration is complete. '
               if linear_certificate else 'Every complete C_B-UNSAT is audited by original dense planner; complete enumerations transferred only after exact affine-space equality checks. '
               if affine_audit else 'Every complete C_B-UNSAT is audited by the original dense planner/kernel. ')
              +'Every SAT carrier Python-audited. High-free supports deferred, never pruned.')
