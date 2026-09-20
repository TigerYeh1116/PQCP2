"""Lightweight exact tests; never run the multi-job wall-clock benchmark."""

from collections import defaultdict
from dataclasses import asdict
import hashlib
from itertools import product
import json
from types import SimpleNamespace

import pytest

from experiments.benchmark_middle_mitm import read_jobs, read_manifest, run_baseline, summarize
from experiments.middle_mitm import (
    bit_signature, decode_middle, generate_middle, make_plan, solve_middle_mitm,
)
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.structured_energy import compress_binary_sequence, fold_profile, target_profile
from solver.cross_compression import compatible_cross_compressions, middle_matches_cross
from solver.target_profiles import TargetContentProfile
from solver.verifier import verify_pqcp


@pytest.mark.parametrize("length", (2,4,6))
def test_bitset_signature_exhaustive_ternary(length):
    for word in product((-1,0,1),repeat=length):
        occupied = sum(1 << i for i,x in enumerate(word) if x)
        negative = sum(1 << i for i,x in enumerate(word) if x < 0)
        assert bit_signature(occupied,negative,length) == pacf(word)[:length//2+1]
        assert decode_middle(occupied,negative,length) == tuple(2*x for x in word)


@pytest.mark.parametrize("n", (1,2,3))
def test_suffix_DP_enumeration_against_all_middle_words(n):
    groups = defaultdict(set)
    for word in product((-1,0,1),repeat=2*n):
        coarse = tuple(2*(word[i]+word[i+n]) for i in range(n))
        key = (coarse,n-sum(word[::2]),n-sum(word[1::2]))
        groups[key].add(word)
    for key, expected in groups.items():
        plan = make_plan(*key)
        counts = plan.norm_counts()
        assert sum(counts.values()) == len(expected)
        generated = [tuple(x//2 for x in decode_middle(o,m,2*n))
                     for o,m in generate_middle(plan,tuple(counts))]
        assert len(generated) == len(set(generated))
        assert set(generated) == expected
        norm = min(counts)
        selected = [tuple(x//2 for x in decode_middle(o,m,2*n))
                    for o,m in generate_middle(plan,(norm,))]
        assert set(selected) == {w for w in expected if sum(x*x for x in w) == norm}


def small_case():
    return (4,), (0,), TargetContentProfile(4,1,1,0,0,1,1)


def test_real_Z3_SAT_is_independently_verified():
    result = solve_middle_mitm(*small_case(),seconds=2)
    assert result["status"] == "SAT" and result["verified"]
    assert verify_pqcp(result["A"],result["B"]).is_valid


@pytest.mark.parametrize("status", ("UNKNOWN","UNSAT"))
def test_exhausted_join_retains_unknown_children(monkeypatch,status):
    calls = []
    def child(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(status=status,construction_time=0,solve_time=0,z3_executed=True)
    monkeypatch.setattr("experiments.middle_mitm.solve_compressed_lift",child)
    result = solve_middle_mitm(*small_case(),seconds=2)
    assert calls
    assert result["status"] == status
    assert result["unknown_children"] == (1 if status == "UNKNOWN" else 0)


def golay_coarse_job():
    # Algebraic compressed fixture, not a previously found PQCP.
    a,b = (1,),(1,)
    for _ in range(4):
        a,b = a+b,a+tuple(-x for x in b)
    return tuple(2*x for x in a)+(0,),tuple(2*x for x in b)+(0,),TargetContentProfile(68,17,-1,12,18,15,15)


def test_index_budget_and_time_budget_are_not_unsat():
    result = solve_middle_mitm(*golay_coarse_job(),seconds=2,max_index_entries=1)
    assert result["status"] == "UNKNOWN" and "index budget" in result["reason"]
    result = solve_middle_mitm(*golay_coarse_job(),seconds=1e-9)
    assert result["status"] == "UNKNOWN"


def test_loader_checks_checksum_and_deduplicates_without_solution_sources(tmp_path):
    a,b,p = golay_coarse_job()
    record = dict(kind="lift",spec=dict(factor=4,compressed_a=a,compressed_b=b,target=asdict(p)))
    checksum = hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":"),
                                        allow_nan=False).encode()).hexdigest()
    text = json.dumps(dict(record=record,sha256=checksum))
    (tmp_path/"a.json").write_text(text)
    (tmp_path/"b.json").write_text(text)
    jobs = read_jobs(tmp_path,8)
    assert len(jobs) == 1 and jobs[0]["id"] == "a"
    saved = tmp_path/"manifest-copy.data"
    saved.write_text(json.dumps(dict(jobs=jobs)))
    assert read_manifest(saved,8) == json.loads(saved.read_text())["jobs"]
    (tmp_path/"b.json").write_text(text.replace(checksum,"broken"))
    with pytest.raises(ValueError,match="checksum"):
        read_jobs(tmp_path,8)


def test_invalid_words_and_content_fail_closed():
    with pytest.raises(ValueError):
        make_plan((3,),0,0)
    with pytest.raises(ValueError):
        make_plan((4,),-1,0)
    assert not make_plan((4,),1,1).norm_counts()


@pytest.mark.parametrize("ca,cb,p", (
    ((4,0,0),(0,0,0),TargetContentProfile(12,3,-1,1,3,2,4)),
    ((2,0),(2,0),TargetContentProfile(8,2,-1,1,2,1,2)),
))
def test_entire_join_and_cross_mask_cache_against_brute_force(monkeypatch,ca,cb,p):
    n = p.L//4
    pools = []
    for c,e,o in ((ca,p.a_even_ones,p.a_odd_ones),(cb,p.b_even_ones,p.b_odd_ones)):
        pools.append([w for w in product((-2,0,2),repeat=2*n)
                      if tuple(w[i]+w[i+n] for i in range(n)) == c
                      and sum(w[::2]) == 2*n-2*e and sum(w[1::2]) == 2*n-2*o])
    cross = compatible_cross_compressions(ca,cb,p.k,p.eta,p) if n%2 else None
    desired = fold_profile(target_profile(p.L,p.k,p.eta),2)
    expected = {(a,b) for a in pools[0] for b in pools[1]
                if tuple(x+y for x,y in zip(pacf(a),pacf(b))) == desired
                and (cross is None or any(middle_matches_cross(a,ra) and middle_matches_cross(b,rb)
                                         for ra,rb in cross))}
    calls = []
    def child(a,b,*args,**kwargs):
        calls.append((a,b))
        return SimpleNamespace(status="UNSAT",construction_time=0,solve_time=0,z3_executed=True)
    monkeypatch.setattr("experiments.middle_mitm.solve_compressed_lift",child)
    result = solve_middle_mitm(ca,cb,p,seconds=2,max_middle_models=10000)
    assert result["status"] == "UNSAT"
    assert set(calls) == expected
    assert len(calls) == len(set(calls))


def test_real_L12_lift_with_cross_constraints():
    a,b = tuple(map(int,"101110111100")),tuple(map(int,"111001110011"))
    p = TargetContentProfile(12,1,1,sum(a[::2]),sum(a[1::2]),sum(b[::2]),sum(b[1::2]))
    result = solve_middle_mitm(compress_binary_sequence(a,4),compress_binary_sequence(b,4),p,2)
    assert result["status"] == "SAT" and verify_pqcp(result["A"],result["B"]).is_valid


def test_censored_aggregation_does_not_invent_reached_times_or_speedup():
    common = dict(verified=False,status="UNKNOWN",budget_seconds=5,measured_wall_seconds=5)
    rows = [dict(common,job="one",method="baseline"),
            dict(common,job="one",method="mitm",first_middle_seconds=.5),
            dict(common,job="two",method="baseline"),
            dict(common,job="two",method="mitm")]
    result = summarize(rows)
    assert result["methods"]["baseline"]["median_first_middle_reached_only"] is None
    assert result["methods"]["mitm"]["median_first_middle_reached_only"] == .5
    assert result["methods"]["mitm"]["first_middle_not_reached"] == 1
    assert result["paired_first_middle"]["mitm_earlier"] == 1
    assert result["paired_first_middle"]["both_not_reached"] == 1
    rows[0]["budget_seconds"] = 10
    with pytest.raises(ValueError,match="budgets"):
        summarize(rows)


@pytest.mark.parametrize("method", ("baseline","mitm"))
def test_same_binary_slice_and_unknown_semantics_for_both_methods(monkeypatch,method):
    limits = []
    def child(*args,**kwargs):
        limits.append(args[4])
        return SimpleNamespace(status="UNKNOWN",construction_time=0,solve_time=0,z3_executed=True)
    if method == "mitm":
        monkeypatch.setattr("experiments.middle_mitm.solve_compressed_lift",child)
        result = solve_middle_mitm(*small_case(),seconds=2,binary_slice_ms=7)
    else:
        monkeypatch.setattr("solver.multilevel_lift.solve_compressed_lift",child)
        result = run_baseline(*small_case(),2,lambda _:None,binary_slice_ms=7)
    assert limits and all(0 < timeout <= 7 for timeout in limits)
    assert result["status"] == "UNKNOWN"
    assert result["binary_history"][0]["status"] == "UNKNOWN"
    assert "middle_a" in result["binary_history"][0]
