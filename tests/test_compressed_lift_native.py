"""Production fast-path uses an exact observed coarse witness, never source bits."""
from unittest.mock import patch

import pytest

from solver.compressed_lift import solve_compressed_lift
from solver.odd_lift_native import solve_odd_lift_native
from solver.target_profiles import TargetContentProfile


A = (2,0,2,2,-2,0,0,0,0,0,0,0,-2,0,0,2,0,0,0,0,0,-2,2,-2,-2,0,2,2,0,0,0,2,0,2,0,-2,2,2,0,-2,2,0,0)
B = (2,0,0,2,0,2,2,2,0,-2,0,0,-2,2,-2,0,2,-2,0,0,2,0,-2,0,2,2,0,2,-2,-2,-2,0,-2,0,0,2,0,2,-2,0,0,2,0)
TARGET = TargetContentProfile(86, 1, -1, 16, 22, 18, 21)


def test_default_exact_native_lift_exhausts_observed_l86_job_without_z3():
    with patch('z3.SolverFor', side_effect=AssertionError('must finish natively')):
        result = solve_compressed_lift(A, B, 1, -1, content_profile=TARGET)
    assert result.status == 'UNSAT' and not result.z3_executed
    assert result.orientation_exact_assignments == 2**21
    assert result.gf2_rank == 22 and result.gf2_free_bits == 21
    old = solve_odd_lift_native(A, B, TARGET, native_preprocessing=False)
    assert old['status'] == 'LIFT_UNSAT' and old['assignments'] == result.orientation_exact_assignments


@pytest.mark.parametrize('arguments', [dict(native_odd_lift=False), dict(excluded_orientations=((0,)*43,))])
def test_rollback_and_blocked_model_queries_keep_general_solver(arguments):
    with patch('solver.odd_lift_native.solve_odd_lift_native', side_effect=AssertionError('ineligible fast path')):
        result = solve_compressed_lift(A, B, 1, -1, 1, TARGET, **arguments)
    assert result.z3_executed


def test_unknown_native_result_falls_through_without_exclusion():
    with patch('solver.odd_lift_native.solve_odd_lift_native', return_value=dict(status='LIFT_UNKNOWN')):
        result = solve_compressed_lift(A, B, 1, -1, 1, TARGET)
    assert result.z3_executed and result.orientation_exact_assignments == 0


def test_invalid_native_flag_rejected():
    with pytest.raises(ValueError, match='native_odd_lift'):
        solve_compressed_lift(A, B, 1, -1, native_odd_lift=1)
