"""Persistence regression using the newly verified experimental solution.

This literal is only a verifier/writer test fixture, never a search seed.
"""

from dataclasses import asdict
import json

import pytest

from experiments.search_middle_mitm_L44 import record_sat
from solver.structured_energy import compress_binary_sequence
from solver.target_profiles import TargetContentProfile, pair_content


def fixture():
    a = tuple(map(int,"11001100000101110000001101001010100001100101"))
    b = tuple(map(int,"11110111100101000100001101100001000100110010"))
    profile = TargetContentProfile(44,2,-1,*pair_content(a,b))
    job = dict(target=asdict(profile),compressed_a=compress_binary_sequence(a,4),
               compressed_b=compress_binary_sequence(b,4))
    return dict(status="SAT",verified=True,A=a,B=b),job


def test_new_solution_persisted_once_and_snapshot_precedes_dedup(tmp_path):
    result,job = fixture()
    output = tmp_path/"snapshots"
    output.mkdir()
    new,payload = record_sat(result,job,tmp_path,output,1)
    assert new
    original = (tmp_path/"44.txt").read_bytes()
    # A/B exchange is also an exact full/compressed target.
    swapped = dict(result,A=result["B"],B=result["A"])
    p = TargetContentProfile(44,2,-1,*pair_content(swapped["A"],swapped["B"]))
    swapped_job = dict(target=asdict(p),compressed_a=job["compressed_b"],compressed_b=job["compressed_a"])
    new,_ = record_sat(swapped,swapped_job,tmp_path,output,2)
    assert not new and (tmp_path/"44.txt").read_bytes() == original
    assert json.loads((output/"sat_000002.json").read_text())["new_solution_written"] is False
    assert payload["nonzero_shifts"] == (2,42)


def test_reject_invalid_candidate_before_any_official_write(tmp_path):
    result,job = fixture()
    result["A"] = (0,)*44
    with pytest.raises(RuntimeError,match="unverified"):
        record_sat(result,job,tmp_path,tmp_path/"snapshots",1)
    assert not (tmp_path/"44.txt").exists()


def test_reject_valid_solution_for_wrong_compression(tmp_path):
    result,job = fixture()
    job["compressed_a"] = (0,)*11
    with pytest.raises(RuntimeError,match="compression"):
        record_sat(result,job,tmp_path,tmp_path/"snapshots",1)
    assert not (tmp_path/"44.txt").exists()
