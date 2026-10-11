"""All modular models, literal integer controls and partial-cap rollback."""
import ctypes
from random import Random

import pytest

from solver.mps_mod8_prefix import quadratic_coefficients
from solver.mod8_model_reference import all_mod8_indices
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.ternary_system_native import _library,prepare_native_system,run_native_system


def model_function(lib):
    fn=lib.pqcp_quadratic_bit_tail_models
    fn.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes[:-2]+[
        ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint64)]
    fn.restype=ctypes.c_int
    return fn


@pytest.mark.parametrize('d',(3,5,7,9,11))
def test_all_modular_models_and_integer_solutions(d):
    rng=Random(866100+d);lib=_library();models=model_function(lib)
    signs=lib.pqcp_quadratic_bit_tail_signs
    signs.argtypes=lib.pqcp_quadratic_hybrid_signs.argtypes;signs.restype=ctypes.c_int
    for case in range(8):
        active=set(rng.sample(range(2*d),d))
        masks=tuple(sum(1<<j for j in range(d) if s*d+j in active) for s in range(2))
        reference=tuple(tuple(rng.choice((-1,1)) if m>>j&1 else 0 for j in range(d)) for m in masks)
        wanted=tuple(x+y for x,y in zip(pacf(reference[0]),pacf(reference[1])))
        negative=tuple(w.count(-1) for w in reference)
        plan=prepare_native_system(masks,d,wanted,negative,conditioning=False)
        assert plan['consistent']
        q=int(plan['meta'][2]);complete=1<<q
        expected=all_mod8_indices(quadratic_coefficients(plan),capacity=complete)
        args=(plan['mask_array'],plan['base'],plan['toggles'],q,d,plan['wanted'],plan['negative'])
        for tail in (1,3,6):
            out=(ctypes.c_uint32*complete)();count=ctypes.c_uint64()
            assert models(*args,tail,complete,out,ctypes.byref(count))==0
            assert tuple(sorted(out[:count.value]))==expected['indices']
            word=(ctypes.c_uint64*2)();stats=(ctypes.c_uint64*7)()
            assert signs(*args,tail,complete,word,stats)==1
            rows=tuple(tuple((-1 if word[s]>>j&1 else 1) if masks[s]>>j&1 else 0 for j in range(d)) for s in range(2))
            assert tuple(r.count(-1) for r in rows)==negative
            assert tuple(x+y for x,y in zip(pacf(rows[0]),pacf(rows[1])))==wanted
            if complete>1:
                old=run_native_system(plan,complete-1,quadratic=True,hybrid_tail=tail)
                status=signs(*args,tail,complete-1,word,stats)
                assert ('UNSAT_FIXED_SUPPORT','SAT_COMPRESSED','UNKNOWN')[status]==old['status']
                assert stats[0]==old['assignments']
                if status==1:
                    rows=tuple(tuple((-1 if word[s]>>j&1 else 1) if masks[s]>>j&1 else 0 for j in range(d)) for s in range(2))
                    assert rows==old['words']


def test_zero_dimensions_and_overflow_count_are_exact():
    lib=_library();fn=model_function(lib)
    for masks,wanted,negative,expected in [((0,0),(0,0,0),(0,0),1),
                                          ((15,0),(4,-1,-1,-1,-1),(2,0),2)]:
        d=len(wanted);p=prepare_native_system(masks,d,wanted,negative,conditioning=False)
        reference=all_mod8_indices(quadratic_coefficients(p),capacity=1<<p['meta'][2])
        out=(ctypes.c_uint32*1)();count=ctypes.c_uint64()
        assert fn(p['mask_array'],p['base'],p['toggles'],p['meta'][2],d,p['wanted'],p['negative'],6,1,out,ctypes.byref(count))==0
        assert count.value==len(reference['indices'])==expected
