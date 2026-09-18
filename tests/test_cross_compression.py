"""Exhaustive, independent checks of CRT margins and exact lift restrictions."""

from itertools import product
from functools import lru_cache
from collections import defaultdict
import json
import random

import pytest

import main
from solver.cross_compression import (
    binary_margins_feasible, compatible_cross_compressions, middle_matches_cross,
    normalize_cross_pairs,
)
from solver.compressed_completion import CompressedLiftJob, run_compressed_lift_job
from solver.compressed_lift import solve_compressed_lift
from solver.multilevel_lift import solve_multilevel_lift
from solver.structured_energy import compress_binary_sequence, fold_profile, target_profile
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def direct_pacf(word):
    return tuple(sum(word[i]*word[(i+u)%len(word)] for i in range(len(word)))
                 for u in range(len(word)))


def direct_compress(bits, factor):
    d = len(bits)//factor
    return tuple(sum(1-2*bits[i] for i in range(r,len(bits),d)) for r in range(d))


def test_all_4_by_3_binary_matrix_margins_against_gale_ryser():
    feasible = set()
    for bits in product((0,1),repeat=12):
        rows = tuple(sum(bits[3*r+c] for c in range(3)) for r in range(4))
        cols = tuple(sum(bits[3*r+c] for r in range(4)) for c in range(3))
        feasible.add((rows,cols))
    # All 32,000 pairs of possible bounded margin vectors, including
    # infeasible ones: verify both no false pruning and no false feasibility.
    for rows in product(range(4),repeat=4):
        for cols in product(range(5),repeat=3):
            assert binary_margins_feasible(rows,cols) == ((rows,cols) in feasible)


def test_equal_totals_are_not_sufficient():
    assert not binary_margins_feasible((3,3,0,0),(4,1,1))
    assert binary_margins_feasible((2,2,1,1),(4,1,1))


def test_L12_all_cross_pairs_for_selected_columns_against_binary_enumeration():
    # Enumerating actual length-12 words provides an independent CRT oracle.
    possible = defaultdict(set)
    for bits in product((0,1),repeat=12):
        possible[direct_compress(bits,4)].add(direct_compress(bits,3))
    rng = random.Random(2026)
    patterns = sorted(possible)
    selected = [((4,0,0),(0,0,0)), ((-4,2,0),(0,2,4))]
    selected += [(rng.choice(patterns),rng.choice(patterns)) for _ in range(25)]
    for ca,cb in selected:
        for k in range(1,6):
            for eta in (-1,1):
                target = tuple(sum(target_profile(12,k,eta)[r::4]) for r in range(4))
                expected = {(a,b) for a in possible[ca] for b in possible[cb]
                            if tuple(x+y for x,y in zip(direct_pacf(a),direct_pacf(b))) == target}
                assert set(compatible_cross_compressions(ca,cb,k,eta)) == expected
                # With fixed parity counts the result must be exactly the
                # corresponding subset, not one guessed row allocation.
                content = TargetContentProfile(12,k,eta,2,2,3,3)
                expected_content = {(a,b) for a,b in expected
                                    if (6-a[0]-a[2])//2 == 2 and (6-a[1]-a[3])//2 == 2
                                    and (6-b[0]-b[2])//2 == 3 and (6-b[1]-b[3])//2 == 3}
                assert set(compatible_cross_compressions(ca,cb,k,eta,content)) == expected_content


def test_L12_all_middle_orientation_feasibility_against_binary_enumeration():
    possible = defaultdict(set)
    for bits in product((0,1),repeat=12):
        possible[direct_compress(bits,2)].add(direct_compress(bits,3))
    for middle in product((-2,0,2),repeat=6):
        for cross in product((-3,-1,1,3),repeat=4):
            assert middle_matches_cross(middle,cross) == (cross in possible[middle])


@pytest.mark.parametrize("L", (12,44,68))
def test_actual_binary_words_never_fail_margin_or_orientation_checks(L):
    rng = random.Random(L)
    n = L//4
    for _ in range(100):
        bits = tuple(rng.randrange(2) for _ in range(L))
        ca, cross = direct_compress(bits,4),direct_compress(bits,n)
        assert binary_margins_feasible(tuple((n-x)//2 for x in cross),
                                       tuple((4-x)//2 for x in ca))
        assert middle_matches_cross(direct_compress(bits,2),cross)


def test_L4_all_cross_constrained_lifts_against_full_enumeration():
    words = list(product((0,1),repeat=4))
    for ca in ((2,2),(0,0),(2,-2)):
        for cb in ((0,0),(2,2),(2,-2)):
            aa = [a for a in words if direct_compress(a,2) == ca]
            bb = [b for b in words if direct_compress(b,2) == cb]
            for eta in (-1,1):
                coarse_a,coarse_b = (sum(ca),),(sum(cb),)
                patterns = compatible_cross_compressions(coarse_a,coarse_b,1,eta)
                # Include an empty disjunction: must not silently disable it.
                for subset in (patterns, patterns[:1], ()):
                    expected = any((direct_compress(a,1),direct_compress(b,1)) in subset
                                   and tuple(x+y for x,y in zip(direct_pacf(direct_compress(a,1)),
                                                                direct_pacf(direct_compress(b,1)))) == target_profile(4,1,eta)
                                   for a in aa for b in bb)
                    result = solve_compressed_lift(ca,cb,1,eta,1_000,cross_compressions=subset)
                    assert result.status in ("SAT","UNSAT")
                    assert (result.status == "SAT") == expected


@lru_cache(maxsize=1)
def _L12_binary_oracle():
    words = tuple(product((0,1),repeat=12))
    return {w: direct_pacf(tuple(1-2*x for x in w)) for w in words}


@pytest.mark.parametrize("k,eta", ((1,-1),(1,1),(2,-1),(2,1),(3,-1),(3,1),(4,-1),(5,-1),(5,1)))
def test_nontrivial_CRT_dimension_preserves_bruteforce_PQCPs(k,eta):
    # Find test witnesses from complete SMALL binary enumeration only.
    # Production pruning never reads these or existing L.txt results.
    oracle = _L12_binary_oracle()
    lookup = {p:w for w,p in oracle.items()}
    desired = target_profile(12,k,eta)
    for a,profile in oracle.items():
        b = lookup.get(tuple(t-x for t,x in zip(desired,profile)))
        if b is not None:
            break
    else:
        pytest.fail("brute-force test case unexpectedly has no witness")
    content = TargetContentProfile(12,k,eta,*pair_content(a,b))
    ca,cb = direct_compress(a,4),direct_compress(b,4)
    pairs = compatible_cross_compressions(ca,cb,k,eta,content)
    assert (direct_compress(a,3),direct_compress(b,3)) in pairs
    result = solve_multilevel_lift(ca,cb,k,eta,2_000,content,max_middle_models=None)
    assert result.status == "SAT" and result.cross_pruning_applied and result.verified
    assert (result.cross_a,result.cross_b) in pairs
    assert pair_content(result.a,result.b) == pair_content(a,b)


def test_cross_precheck_avoids_z3_for_incompatible_patterns(monkeypatch):
    # Coarse norms match fold_4(T), but row marginals/content have no completion.
    ca,cb = (4,), (0,)
    assert (16,) == fold_profile(target_profile(4,1,1),4)
    content = TargetContentProfile(4,1,1,1,0,1,1)
    monkeypatch.setattr("z3.SolverFor",lambda *_a: pytest.fail("should prune before solver"))
    result = solve_multilevel_lift(ca,cb,1,1,1_000,content)
    assert result.status == "UNSAT" and result.cross_pruning_applied
    assert result.compatible_cross_pairs == 0 and not result.z3_executed


def test_saved_cross_metadata_and_verified_output(tmp_path):
    a,b = (0,0,0,0),(0,0,1,1)
    target = TargetContentProfile(4,1,1,*pair_content(a,b))
    job = CompressedLiftJob((4,),(0,),target,a,b,4)
    result = run_compressed_lift_job(job,1_000,tmp_path,123)
    snapshot = json.loads(open(result["snapshot"]).read())
    assert result["cross_pruning_applied"] and result["compatible_cross_pairs"] > 0
    assert snapshot["algorithm_parameters"]["cross_compression_pruning"]
    assert snapshot["cross_pattern_count"] > 0
    assert snapshot["cross_pruning_applied"] and verify_pqcp(snapshot["A"],snapshot["B"]).is_valid
    assert tuple(snapshot["cross_a"]) == direct_compress(snapshot["A"],1)
    assert tuple(snapshot["cross_b"]) == direct_compress(snapshot["B"],1)
    assert (tmp_path/"4.txt").exists()


def test_non_coprime_falls_back_and_rollback_is_available(tmp_path):
    a,b = tuple(map(int,"00000111")),tuple(map(int,"00010011"))
    result = solve_multilevel_lift(direct_compress(a,4),direct_compress(b,4),2,-1,1_000)
    assert result.status == "SAT" and not result.cross_pruning_applied
    result = solve_multilevel_lift((4,),(0,),1,1,1_000,cross_compression_pruning=False)
    assert result.status == "SAT" and not result.cross_pruning_applied
    assert main.parse_args(["--L","68"]).z3_cross_compression_pruning
    args = main.parse_args(["--L","68","--no-z3-cross-compression-pruning"])
    assert not args.z3_cross_compression_pruning


@pytest.mark.parametrize("ca,cb", (((4,0),(0,0)),((1,),(0,)),((True,),(0,)),((0,),(0,0,0))))
def test_invalid_cross_inputs_rejected(ca,cb):
    with pytest.raises(ValueError): compatible_cross_compressions(ca,cb,1,1)


def test_invalid_cross_constraints_rejected():
    with pytest.raises(ValueError): normalize_cross_pairs([((3,3,3,3),(3,3,3,3))],12,1,1)
    with pytest.raises(ValueError): normalize_cross_pairs([],8,1,1)
