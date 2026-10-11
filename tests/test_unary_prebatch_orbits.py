from itertools import product
import base64
import struct

import pytest

from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.full_row_completion import unary_sign_completion
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.unary_norm_batch import complete_unary_norm_batch, CompleteUnarySupportOrbitUnsatCache


def test_complete_single_norm_exclusion_prunes_rotated_support_before_batch():
    cache = CompleteUnarySupportOrbitUnsatCache()
    args = (5, (2, 0, 0, 0, 0), 1, 2**20)
    a = complete_unary_norm_batch((3,), *args, dihedral_unsat_cache=cache)
    b = complete_unary_norm_batch((6,), *args, dihedral_unsat_cache=cache)
    assert a['native_input_supports'] == 1 and a['unsat_supports'] == 1
    assert b['native_input_supports'] == 0 and b['dihedral_unsat_supports'] == 1
    assert b['unsat_supports'] == 1 and not b['candidates']
    receipts=list(struct.iter_unpack('=QQQQQQ',base64.b64decode(b['dihedral_unsat_receipts_base64'])))
    assert receipts[0][1:3] == (6,3) and b['certificate_rows']==0
    with pytest.raises(ValueError, match='different single norm/count'):
        complete_unary_norm_batch((6,), 5, args[1], 0, 2**20, dihedral_unsat_cache=cache)


def test_sat_and_deferred_supports_are_never_excluded_by_orbit_cache():
    profile = tuple(pacf((1, -1, 0, 0, 0)))
    for free in (0, 12):
        cache = CompleteUnarySupportOrbitUnsatCache()
        for mask in (3, 6):
            out = complete_unary_norm_batch((mask,), 5, profile, 1, 2**20,
                                           free_limit=free, dihedral_unsat_cache=cache)
            assert out['native_input_supports'] == 1 and out['dihedral_unsat_supports'] == 0
            assert out['sat_supports']+out['deferred_supports'] == 1
        assert not cache._proofs


def test_full_solver_proof_reuse_requires_complete_single_support_unsat():
    cache = CompleteUnarySupportOrbitUnsatCache()
    wanted = (2, 0, 0, 0, 0)
    cache.bind((5, wanted, 1))
    raw = unary_sign_completion(3, 5, wanted, 1, 2**20)
    assert raw['status'] == 'UNSAT_FIXED_SUPPORT'
    proof = dict(raw, complete=True)  # Terminal scalar status supplies this fact.
    cache.remember_full_exclusion(3, proof, 5)
    out = complete_unary_norm_batch((6,), 5, wanted, 1, 2**20, dihedral_unsat_cache=cache)
    assert out['native_input_supports'] == 0
    assert out['dihedral_unsat_full_source_results']['3']['complete']
    for bad in ({'status': 'UNKNOWN', 'complete': False}, {'status': 'SAT_COMPRESSED', 'complete': False}):
        with pytest.raises(ValueError):
            cache.remember_full_exclusion(3, bad, 5)


def test_prebatch_quotient_preserves_every_small_complete_decision_and_first_model():
    for row in product((-1, 0, 1), repeat=3):
        for k, eta in product((1, 2), (-1, 1)):
            args = dict(k=k, eta=eta, unary_batch_free=12, dihedral_support=True)
            a = complete_fixed_coarse_row_target(row, **args)
            b = complete_fixed_coarse_row_target(row, prebatch_dihedral=True, **args)
            assert (a['status'], a['complete'], a['words']) == (b['status'], b['complete'], b['words'])


def test_bulk_dihedral_keys_equal_every_small_literal_support():
    from solver.cyclic_support_native import dihedral_support_key,dihedral_support_keys
    for d in (3,5,11):
        masks=tuple(range(1<<d))
        assert dihedral_support_keys(masks,d)==tuple(dihedral_support_key(m,d) for m in masks)
