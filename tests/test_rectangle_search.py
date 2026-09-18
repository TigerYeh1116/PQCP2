"""Exact margins/deltas, real C search, bounded restart and Z3 handoff tests."""

from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import random

import pytest

import main
from solver.compressed_completion import CompressedLiftJob
from solver.compressed_lift import solve_compressed_lift
from solver.correlation import full_correlation_profile, periodic_autocorrelation
from solver.cross_compression import binary_margins_feasible
from solver.rectangle_search import (crt_position, initialize_from_margins,
                                     rectangle_correlation_delta, search_rectangles)
from solver.staged_completion import CompletionWorkStore, SlicePolicy
from solver.structured_energy import compress_binary_sequence, full_error_energy
from solver.target_profiles import TargetContentProfile, pair_content
from solver.verifier import verify_pqcp


def job(L=12):
    # Deterministic random-source exact matches, NOT previously found PQCPs.
    a,b,k,eta = {
        12: ("101110111100","111001110011",1,1),
        20: ("10101011100110011101","11100001010011011111",2,-1),
        44: ("10111100001011110001100111101101101010101101",
             "00101001010010011001111101100100010111001000",4,-1),
    }[L]
    a,b = tuple(map(int,a)),tuple(map(int,b))
    return CompressedLiftJob(compress_binary_sequence(a,4),compress_binary_sequence(b,4),
                             TargetContentProfile(L,k,eta,*pair_content(a,b)),a,b,4)


@pytest.mark.parametrize("n", (1,3,5,11,17,23))
def test_crt_layout(n):
    indices = [crt_position(n,r,c) for r in range(4) for c in range(n)]
    assert sorted(indices) == list(range(4*n))
    for r in range(4):
        for c in range(n):
            i = crt_position(n,r,c)
            assert i%4 == r and i%n == c


def test_initializer_exhaustive_all_labeled_4_by_3_margins():
    feasible = set()
    for bits in product((0,1),repeat=12):
        feasible.add((tuple(sum(bits[i] for i in range(12) if i%4==r) for r in range(4)),
                      tuple(sum(bits[i] for i in range(12) if i%3==c) for c in range(3))))
    for rows in product(range(4),repeat=4):
        for cols in product(range(5),repeat=3):
            expected = (rows,cols) in feasible
            assert binary_margins_feasible(rows,cols) == expected
            if expected:
                ca, cross = tuple(4-2*x for x in cols),tuple(3-2*x for x in rows)
                bits = initialize_from_margins(ca,cross,123)
                assert compress_binary_sequence(bits,4) == ca
                assert compress_binary_sequence(bits,3) == cross


@pytest.mark.parametrize("L", (12,20,44,68))
def test_random_consecutive_rectangles_exact_delta_and_all_invariants(L):
    rng = random.Random(220+L)
    bits = tuple(rng.randrange(2) for _ in range(L))
    n = L//4
    fixed = (compress_binary_sequence(bits,4),compress_binary_sequence(bits,n),
             sum(bits),sum(bits[::2]),sum(bits[1::2]))
    for _ in range(80):
        legal = []
        for r,s in combinations(range(4),2):
            for c,d in combinations(range(n),2):
                p = tuple(crt_position(n,row,col) for row,col in ((r,c),(r,d),(s,c),(s,d)))
                if bits[p[0]] == bits[p[3]] != bits[p[1]] == bits[p[2]]:
                    legal.append((r,s,c,d,p))
        assert legal
        r,s,c,d,p = rng.choice(legal)
        delta = rectangle_correlation_delta(bits,(r,s),(c,d))
        after = tuple(bit ^ int(i in p) for i,bit in enumerate(bits))
        assert delta == tuple(periodic_autocorrelation(after,u)-periodic_autocorrelation(bits,u)
                              for u in range(L))
        assert delta[0] == 0 and all(delta[u] == delta[-u % L] for u in range(L))
        assert (compress_binary_sequence(after,4),compress_binary_sequence(after,n),
                sum(after),sum(after[::2]),sum(after[1::2])) == fixed
        bits = after


def test_all_L12_checkerboards_have_exact_C_deltas():
    # Fixed rectangle, all assignments of the other eight bits and both
    # checkerboard orientations. Includes edges between two flipped corners.
    p = tuple(crt_position(3,r,c) for r,c in ((0,0),(0,2),(3,0),(3,2)))
    free = [i for i in range(12) if i not in p]
    for values in product((0,1),repeat=9):
        bits = [0]*12
        for i,bit in zip(free,values[:8]): bits[i] = bit
        for i,bit in zip(p,(values[8],1-values[8],1-values[8],values[8])): bits[i] = bit
        after = [bit ^ int(i in p) for i,bit in enumerate(bits)]
        assert rectangle_correlation_delta(bits,(0,3),(0,2)) == tuple(
            periodic_autocorrelation(after,u)-periodic_autocorrelation(bits,u) for u in range(12))


@pytest.mark.parametrize("L", (12,44,68))
def test_C_many_moves_match_full_recomputation_and_repeat_deterministically(L):
    rng = random.Random(L)
    a,b = (tuple(rng.randrange(2) for _ in range(L)) for _ in range(2))
    target = TargetContentProfile(L,1,-1,*pair_content(a,b))
    first = search_rectangles(a,b,target,123,moves=300,samples=6)
    again = search_rectangles(a,b,target,123,moves=300,samples=6)
    assert first.a == again.a and first.b == again.b
    assert (first.moves,first.evaluations,first.accepted) == (again.moves,again.evaluations,again.accepted)
    assert first.profile == tuple(full_correlation_profile(first.a,first.b))
    assert first.best_energy == full_error_energy(first.profile,1,-1) <= first.initial_energy
    assert first.evaluations == 6*first.moves and first.moves <= 300


def test_no_legal_rectangle_is_a_miss_not_unsat():
    a,b = (0,)*12,(1,)*12
    result = search_rectangles(a,b,TargetContentProfile(12,1,1,*pair_content(a,b)),123)
    assert result.no_legal_moves and not result.verified
    assert result.moves == result.evaluations == 0 and result.best_energy > 0


def test_infeasible_margins_and_illegal_rectangles_rejected():
    with pytest.raises(ValueError,match="infeasible"):
        initialize_from_margins((4,4,4),(-3,3,3,3),1)
    with pytest.raises(ValueError,match="checkerboard"):
        rectangle_correlation_delta((0,)*12,(0,1),(0,1))
    with pytest.raises(ValueError):
        crt_position(4,0,0)
    with pytest.raises(ValueError):
        rectangle_correlation_delta((0,)*12,(0,0),(0,1))


def test_rectangle_solution_saved_without_falsely_counting_Z3(tmp_path):
    original = job()
    assert not verify_pqcp(original.source_a,original.source_b).is_valid
    store = CompletionWorkStore(tmp_path,12,123,SlicePolicy(rectangle_moves=20))
    key = store.register("lift",asdict(original))
    report = store.advance_lift(key)
    assert report["status"] == "RECTANGLE_SAT" and report["solved"]
    assert not report["z3_executed"] and report["z3_calls"] == 0
    data = json.loads(Path(report["snapshot"]).read_text())
    assert data["method"] == "dual_compression_rectangle"
    assert verify_pqcp(data["A"],data["B"]).is_valid
    assert (tmp_path/"12.txt").exists()
    assert store.read(key)["phase"] == "SAT"
    assert store.read(key)["state"]["rectangle"]["best_energy"] == 0


def test_bounded_misses_restart_patterns_then_handoff_to_Z3(tmp_path):
    policy = SlicePolicy(rectangle_moves=1,rectangle_samples=1,rectangle_restarts=4)
    store = CompletionWorkStore(tmp_path,20,123,policy)
    key = store.register("lift",asdict(job(20)))
    selected = []
    for restart in range(1,5):
        report = store.advance_lift(key)
        assert report["status"] == "RECTANGLE_MISS" and report["phase"] == "READY"
        assert report["rectangle_restart"] == restart and report["z3_calls"] == 0
        selected.append(report["rectangle_pattern"])
        assert store.read(key)["state"]["children"] == []
    assert len(set(selected)) == 4
    report = store.advance_lift(key)
    assert report["work_kind"] == "enum" and report["z3_calls"] == 1
    if report["phase"] == "READY" and store.read(key)["state"]["children"]:
        child = store.advance_lift(key)
        assert child["preferred_values"] == 10  # one orientation per compressed zero


def test_resume_keeps_pattern_order_and_best_bits(tmp_path):
    policy = SlicePolicy(rectangle_moves=1,rectangle_samples=1,rectangle_restarts=4)
    continuous = CompletionWorkStore(tmp_path/"continuous",20,123,policy)
    split = CompletionWorkStore(tmp_path/"split",20,123,policy)
    key = continuous.register("lift",asdict(job(20)))
    assert split.register("lift",asdict(job(20))) == key
    for _ in range(4): continuous.advance_lift(key)
    for _ in range(2): split.advance_lift(key)
    restored = CompletionWorkStore(tmp_path/"split",20,123,policy)
    assert restored.recover() == [key]
    for _ in range(2): restored.advance_lift(key)
    assert restored.read(key)["state"] == continuous.read(key)["state"]


def test_Z3_preferences_are_not_hard_bit_fixes():
    a,b = tuple(map(int,"00000111")),tuple(map(int,"00010011"))
    target = TargetContentProfile(8,2,-1,*pair_content(a,b))
    result = solve_compressed_lift(compress_binary_sequence(a,2),compress_binary_sequence(b,2),
                                   2,-1,2000,target,preferred_pair=((0,)*8,(0,)*8))
    assert result.status == "SAT" and result.verified and result.preferred_values == 4
    assert result.a != (0,)*8 or result.b != (0,)*8


def test_disable_rectangle_keeps_original_staged_Z3(tmp_path):
    store = CompletionWorkStore(tmp_path,12,123,SlicePolicy(rectangle_search=False))
    key = store.register("lift",asdict(job()))
    assert store.advance_lift(key)["work_kind"] == "enum"
    assert "rectangle" not in store.read(key)["state"]
    args = main.parse_args(["--L","68","--no-z3-rectangle-search"])
    assert not args.z3_rectangle_search


@pytest.mark.parametrize("kwargs", ({"rectangle_moves":0},{"rectangle_restarts":0},
                                    {"rectangle_samples":0},{"rectangle_samples":1025}))
def test_rectangle_policy_validation(kwargs):
    with pytest.raises(ValueError): SlicePolicy(**kwargs)
