"""Prove small orbit/fibre classifications, and check queue/resume admission."""

from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product
from math import gcd
import random

import pytest

import main
from solver.compressed_completion import CompressedLiftJob, CompressedLiftPool
from solver.compressed_equivalence import compressed_job_equivalence_key
from solver.correlation import full_correlation_profile, periodic_autocorrelation
from solver.solution_equivalence import pqcp_equivalence_key
from solver.staged_completion import CompletionWorkStore, SlicePolicy
from solver.structured_energy import compress_binary_sequence, fold_profile, target_profile
from solver.target_profiles import TargetContentProfile, pair_content


A = tuple(map(int, "00000111"))
B = tuple(map(int, "00010011"))


def make_job(a=A, b=B, k=2, eta=-1, factor=2):
    return CompressedLiftJob(compress_binary_sequence(a, factor),
                             compress_binary_sequence(b, factor),
                             TargetContentProfile(len(a), k, eta, *pair_content(a, b)),
                             a, b, factor)


def transform_bits(word, q, reverse, offset, complement):
    L = len(word)
    return tuple(word[((-1 if reverse else 1)*q*i+offset) % L] ^ complement
                 for i in range(L))


def transformed_job(job, q=1, ra=0, rb=0, ta=0, tb=0, na=0, nb=0, swap=False):
    a = transform_bits(job.source_a, q, ra, ta, na)
    b = transform_bits(job.source_b, q, rb, tb, nb)
    k = pow(q, -1, job.target.L) * job.target.k % job.target.L
    if swap:
        a, b = b, a
    return make_job(a, b, min(k, job.target.L-k), job.target.eta, job.factor)


def descriptor(job):
    return (job.compressed_a, job.compressed_b, job.target)


@pytest.mark.parametrize("L,factor", ((4,2), (4,4), (6,2)))
def test_all_small_subproblem_orbits_match_explicit_full_sequence_actions(L, factor):
    """Both directions: every explicit orbit is ONE key; no distinct orbits merge."""
    words = tuple(product((0,1), repeat=L))
    groups = defaultdict(set)
    representatives = {}
    for a in words:
        for b in words:
            for k in range(1, L//2):
                for eta in (-1, 1):
                    job = make_job(a, b, k, eta, factor)
                    key = job.canonical_key
                    groups[key].add(descriptor(job))
                    representatives.setdefault(key, job)
    for key, group in groups.items():
        job = representatives[key]
        explicit = set()
        for q in range(1, L):
            if gcd(q, L) != 1:
                continue
            mapped = pow(q, -1, L) * job.target.k % L
            mapped = min(mapped, L-mapped)
            sides = []
            for word in (job.source_a, job.source_b):
                row_orbit = set()
                for reverse, offset, complement in product((0,1), range(L), (0,1)):
                    full = transform_bits(word, q, reverse, offset, complement)
                    row_orbit.add((compress_binary_sequence(full, factor),
                                   sum(full[::2]), sum(full[1::2])))
                sides.append(row_orbit)
            for left, right in product(*sides):
                for aa, bb in ((left, right), (right, left)):
                    explicit.add((aa[0], bb[0], TargetContentProfile(
                        L, mapped, job.target.eta, aa[1], aa[2], bb[1], bb[2])))
        assert group == explicit


@pytest.mark.parametrize("L,factor", ((4,2), (4,4), (6,2), (8,2), (8,4)))
def test_exhaustive_binary_lifts_preserve_solution_counts_and_solution_orbits(L, factor):
    """All binary pairs, including empty fibres: same key => same lift solutions up to symmetry."""
    words = tuple(product((0,1), repeat=L))
    pacf = {w: tuple(periodic_autocorrelation(w,u) for u in range(L)) for w in words}
    compressed = {w: compress_binary_sequence(w, factor) for w in words}
    content = {w: (sum(w[::2]), sum(w[1::2])) for w in words}
    fibres = defaultdict(list)
    for a, b in product(words, repeat=2):
        fibres[(compressed[a], compressed[b], content[a]+content[b])].append((a,b))
    orbit_results = {}
    for (ca, cb, counts), pairs in fibres.items():
        # Compare all prescribed target cases, not just fibres with solutions.
        by_profile = defaultdict(list)
        for a,b in pairs:
            by_profile[tuple(x+y for x,y in zip(pacf[a],pacf[b]))].append((a,b))
        for k in range(1,L//2):
            for eta in (-1,1):
                target = TargetContentProfile(L,k,eta,*counts)
                key = compressed_job_equivalence_key(ca,cb,target,factor)
                solutions = by_profile.get(target_profile(L,k,eta), ())
                summary = (len(solutions), frozenset(pqcp_equivalence_key(a,b) for a,b in solutions))
                assert orbit_results.setdefault(key, summary) == summary
    assert any(count for count, _ in orbit_results.values())
    assert any(not count for count, _ in orbit_results.values())


@pytest.mark.parametrize("L,factor", ((12,4), (44,2), (44,4), (46,2), (68,2), (68,4)))
def test_large_transformations_carry_full_target_and_parity_content(L, factor):
    rng = random.Random(300+L+factor)
    units = [q for q in range(1,L) if gcd(q,L)==1]
    for _ in range(6):
        a, b = (tuple(rng.randrange(2) for _ in range(L)) for _ in range(2))
        original = make_job(a,b,rng.randrange(1,L//2),rng.choice((-1,1)),factor)
        q = rng.choice(units)
        new = transformed_job(original, q, *[rng.randrange(2) for _ in range(2)],
                              *[rng.randrange(L) for _ in range(2)],
                              *[rng.randrange(2) for _ in range(2)], swap=True)
        assert new.canonical_key == original.canonical_key
        profile = full_correlation_profile(a,b)
        assert full_correlation_profile(new.source_a,new.source_b) == [profile[q*u % L] for u in range(L)]


@pytest.mark.parametrize("L,factor", ((6,2), (44,4), (46,2), (68,4)))
def test_odd_compressed_length_hidden_rotation_swaps_content(L, factor):
    a = (1,)+(0,)*(L-1)
    original = make_job(a,(0,)*L,1,-1,factor)
    moved = transformed_job(original, ta=L//factor)
    assert moved.compressed_a == original.compressed_a
    assert moved.target.a_even_ones != original.target.a_even_ones
    assert moved.canonical_key == original.canonical_key


def test_even_compressed_length_cannot_swap_content_without_transforming_pattern():
    original = make_job()
    wrong = replace(original, target=replace(original.target, a_even_ones=2, a_odd_ones=1))
    assert wrong.canonical_key != original.canonical_key


def test_target_sign_shift_class_and_factor_are_not_dropped_from_key():
    original = make_job()
    assert replace(original, target=replace(original.target, eta=1)).canonical_key != original.canonical_key
    assert replace(original, target=replace(original.target, k=1)).canonical_key != original.canonical_key
    assert make_job(factor=4).canonical_key != original.canonical_key


def test_equal_folded_target_does_not_replace_full_target_position():
    job = make_job((0,)*12,(1,)*12,1,-1,4)
    other = replace(job,target=replace(job.target,k=4))
    assert fold_profile(target_profile(12,1,-1),4) == fold_profile(target_profile(12,4,-1),4)
    assert job.canonical_key != other.canonical_key


def test_equal_pacf_signature_is_not_an_equivalence_key():
    a,b,c,d = (tuple(map(int,s)) for s in ("000011","001011","000111","001001"))
    assert full_correlation_profile(a,b) == full_correlation_profile(c,d)
    assert make_job(a,b,1,-1).canonical_key != make_job(c,d,1,-1).canonical_key


def test_independent_nontrivial_decimation_is_not_allowed():
    rng = random.Random(347)
    a,b = (tuple(rng.randrange(2) for _ in range(44)) for _ in range(2))
    original = make_job(a,b,1,-1)
    wrong = make_job(transform_bits(a,3,0,0,0),b,1,-1)
    assert wrong.canonical_key != original.canonical_key


def test_pool_admits_only_one_equivalent_job_but_keeps_original_coordinates():
    pool = CompressedLiftPool(8)
    jobs = pool.offer(A,B,2,-1)
    assert jobs and len({job.canonical_key for job in jobs}) == len(jobs)
    for job in jobs:
        pool.submit(job)
    for kwargs in ({"swap":True}, {"na":1}, {"ta":3,"rb":1}, {"q":3}):
        new = transformed_job(make_job(), **kwargs)
        assert not pool.offer(new.source_a,new.source_b,new.target.k,new.target.eta)
    assert pool.duplicates_skipped > 0


def test_store_deduplicates_unknown_and_resume_retains_original_state(tmp_path):
    store = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    first = make_job(factor=4)
    key = store.register("lift",asdict(first))
    store.advance_lift(key)  # a real middle model, not a full solution
    before = store.path(key).read_bytes()
    equivalent = transformed_job(first, ta=3, na=1, swap=True)
    assert store.register("lift",asdict(equivalent)) is None
    assert store.path(key).read_bytes() == before
    restored = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    assert restored.recover() == [key]
    record = restored.read(key)
    assert record["state"]["children"] and record["state"]["order"][0] == 0
    assert tuple(record["spec"]["compressed_a"]) == first.compressed_a
    assert tuple(record["spec"]["source_a"]) == A
    assert restored.register("lift",asdict(equivalent)) is None


def test_legacy_duplicate_files_preserved_and_progressed_representative_selected(tmp_path):
    old = CompletionWorkStore(tmp_path,8,123,SlicePolicy(),deduplicate_equivalents=False)
    original = make_job(factor=4)
    moved = transformed_job(original, na=1, swap=True)
    first = old.register("lift",asdict(original))
    second = old.register("lift",asdict(moved))
    old.advance_lift(second)
    first_bytes = old.path(first).read_bytes()
    prior = old.read(second)
    new = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    assert new.recover() == [second]
    assert new.recovery_duplicates == 1
    assert old.path(first).read_bytes() == first_bytes
    assert new.read(second)["state"]["children"] == prior["state"]["children"]
    assert new.read(second)["spec"] == prior["spec"]
    # No files deleted or aliases marked UNSAT; rollback can still see both.
    assert set(old.recover()) == {first,second}


def test_legacy_verified_sat_prevents_requeue_of_pending_equivalent(tmp_path):
    old = CompletionWorkStore(tmp_path,8,123,SlicePolicy(),deduplicate_equivalents=False)
    first = old.register("lift",asdict(make_job()))
    assert old.advance_lift(first)["solved"]
    second = old.register("lift",asdict(transformed_job(make_job(), swap=True)))
    untouched = old.path(second).read_bytes()
    new = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    assert new.recover() == []
    assert old.path(second).read_bytes() == untouched
    assert new.register("lift",asdict(transformed_job(make_job(), na=1))) is None


def test_conflicting_legacy_proofs_fail_without_overwriting(tmp_path):
    old = CompletionWorkStore(tmp_path,8,123,SlicePolicy(),deduplicate_equivalents=False)
    first = old.register("lift",asdict(make_job()))
    second = old.register("lift",asdict(transformed_job(make_job(), swap=True)))
    for key,phase in ((first,"SAT"),(second,"UNSAT")):
        record = old.read(key)
        record["phase"] = phase
        old.save(key,record)
    before = {key:old.path(key).read_bytes() for key in (first,second)}
    new = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    with pytest.raises(ValueError,match="disagree"):
        new.recover()
    assert {key:old.path(key).read_bytes() for key in (first,second)} == before


def test_duplicate_checkpoint_corruption_is_not_hidden_by_dedup(tmp_path):
    old = CompletionWorkStore(tmp_path,8,123,SlicePolicy(),deduplicate_equivalents=False)
    first = old.register("lift",asdict(make_job()))
    second = old.register("lift",asdict(transformed_job(make_job(), swap=True)))
    old.path(second).write_text("{broken")
    before = old.path(first).read_bytes()
    with pytest.raises(ValueError,match="invalid completion checkpoint"):
        CompletionWorkStore(tmp_path,8,123,SlicePolicy()).recover()
    assert old.path(first).read_bytes() == before


def test_radius_centres_are_not_compressed_equivalence_jobs(tmp_path):
    store = CompletionWorkStore(tmp_path,8,123,SlicePolicy())
    first = store.register("radius",{"a":A,"b":B,"encoded":(),"parameters":{},"score":10})
    second = store.register("radius",{"a":B,"b":A,"encoded":(),"parameters":{},"score":10})
    assert first and second and first != second
    assert set(store.recover()) == {first,second}


def test_default_bridge_queues_one_equivalent_fixed_problem_before_z3(tmp_path, monkeypatch):
    # Do not run any solver: inspect real default admission and disk queue.
    monkeypatch.setattr(main._CCompletionBridge,"_worker",lambda self: None)
    args = main.parse_args(["--L","8","--z3-trigger-score","0"])
    bridge = main._CCompletionBridge(args,8,tmp_path)
    try:
        for params in ({}, {"swap":True}, {"na":1}, {"ta":1}, {"q":3}):
            job = transformed_job(make_job(), **params)
            bridge.consider(job.source_a,job.source_b,{"k":2,"sign":-1,"score":100})
        records = [bridge._staged_store.read(p.stem)
                   for p in bridge._staged_store.directory.glob("*.json")]
        assert sorted(r["spec"]["factor"] for r in records) == [2,4]
        assert bridge._queue.qsize() == 2 and bridge.z3_execution_count == 0
    finally:
        bridge.close()


def test_invalid_compressed_key_inputs_are_rejected():
    original = make_job()
    for a,b,target,factor in (((),original.compressed_b,original.target,2),
                              ((1,0,0,0),original.compressed_b,original.target,2),
                              (original.compressed_a,original.compressed_b,original.target,3)):
        with pytest.raises(ValueError):
            compressed_job_equivalence_key(a,b,target,factor)
