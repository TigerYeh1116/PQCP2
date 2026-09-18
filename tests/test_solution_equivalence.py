"""Orbit keys, shared writer, and backed-up cleanup correctness."""

from itertools import product
from math import gcd
from pathlib import Path
import random

import pytest

from experiments.deduplicate_solutions import plan_cleanup, apply_cleanup
from solver.search_runner import append_verified_solution_if_new, SearchImplementationError
from solver.solution_equivalence import pqcp_equivalence_key
from solver.verifier import verify_pqcp


def row_orbit(s):
    out = set()
    for word in (s, s[::-1]):
        for i in range(len(s)):
            rotated = word[i:] + word[:i]
            out.add(rotated)
            out.add("".join("1" if x == "0" else "0" for x in rotated))
    return out


def test_exact_group_key_against_explicit_all_L4_pair_orbits():
    words = ["".join(map(str, w)) for w in product((0, 1), repeat=4)]
    unseen = set(product(words, repeat=2))
    keys = set()
    while unseen:
        a, b = next(iter(unseen))
        key = pqcp_equivalence_key(a, b)
        assert key not in keys  # no merging two different explicit orbits
        keys.add(key)
        orbit = set()
        for q in (1, 3):
            aa = "".join(a[q * i % 4] for i in range(4))
            bb = "".join(b[q * i % 4] for i in range(4))
            for x in row_orbit(aa):
                for y in row_orbit(bb):
                    orbit.update(((x, y), (y, x)))
        assert all(pqcp_equivalence_key(x, y) == key for x, y in orbit)
        unseen.difference_update(orbit)


@pytest.mark.parametrize("L", (1, 4, 6, 8, 44, 46, 68))
def test_composed_transformations_keep_key(L):
    rng = random.Random(100 + L)
    a = "".join(str(rng.randrange(2)) for _ in range(L))
    b = "".join(str(rng.randrange(2)) for _ in range(L))
    key = pqcp_equivalence_key(a, b)
    units = [q for q in range(1, L + 1) if gcd(q, L) == 1]
    for _ in range(8):
        q = rng.choice(units)
        transformed = []
        for word in (a, b):
            word = "".join(word[q * i % L] for i in range(L))
            if rng.randrange(2): word = word[::-1]
            offset = rng.randrange(L)
            word = word[offset:] + word[:offset]
            if rng.randrange(2): word = "".join("1" if x == "0" else "0" for x in word)
            transformed.append(word)
        assert pqcp_equivalence_key(*transformed[::-1]) == key


def test_equal_pair_PACF_does_not_collapse_distinct_orbits():
    # Both L6 pairs are verified with identical pair profiles but different
    # individual row structure. PACF alone must NOT be a deduplication key.
    a, b = "000011", "001011"
    c, d = "000111", "001001"
    assert verify_pqcp(a, b).is_valid and verify_pqcp(c, d).is_valid
    assert verify_pqcp(a, b).profile == verify_pqcp(c, d).profile
    assert pqcp_equivalence_key(a, b) != pqcp_equivalence_key(c, d)


def test_writer_rejects_full_symmetry_duplicate_and_preserves_format(tmp_path):
    assert append_verified_solution_if_new(4, "0000", "0011", tmp_path)
    old = (tmp_path / "4.txt").read_bytes()
    for a in row_orbit("0000"):
        for b in row_orbit("0011"):
            assert not append_verified_solution_if_new(4, b, a, tmp_path)
    assert (tmp_path / "4.txt").read_bytes() == old
    assert append_verified_solution_if_new(4, "0011", "0101", tmp_path)


def test_writer_index_refreshes_after_external_changes(tmp_path):
    path = tmp_path / "4.txt"
    assert append_verified_solution_if_new(4, "0000", "0011", tmp_path)
    path.write_text("")
    assert append_verified_solution_if_new(4, "0000", "0110", tmp_path)
    assert not append_verified_solution_if_new(4, "1111", "1001", tmp_path)


def test_wrong_destination_length_and_malformed_records_are_rejected(tmp_path):
    with pytest.raises(SearchImplementationError, match="length"):
        append_verified_solution_if_new(6, "0000", "0011", tmp_path)
    (tmp_path / "4.txt").write_text("a=0000\nb=0022\n")
    with pytest.raises(SearchImplementationError, match="malformed"):
        append_verified_solution_if_new(4, "0000", "0011", tmp_path)


def block(a, b):
    check = verify_pqcp(a, b)
    assert check.is_valid
    return "L={}\nnonzero shifts={}\nnonzero PACS={}\na={}\nb={}\n\n".format(
        len(a), ",".join(map(str, check.nonzero_shifts)),
        check.profile[check.nonzero_shifts[0]], a, b)


def test_cleanup_preserves_first_exact_block_and_backup(tmp_path):
    first = block("0000", "0011")
    distinct = block("0011", "0101")
    original = "\n" + first + block("1111", "0110") + distinct
    path = tmp_path / "4.txt"
    path.write_text(original)
    plan = plan_cleanup(path)
    assert plan.original_count == 3 and plan.retained_count == 2
    assert plan.cleaned == "\n" + first + distinct
    backup = tmp_path / "backup"
    apply_cleanup([plan], backup)
    assert (backup / "4.txt").read_bytes() == original.encode()
    assert path.read_text() == plan.cleaned
    assert plan_cleanup(path).removed_count == 0
    unchanged = path.read_bytes()
    apply_cleanup([plan_cleanup(path)], tmp_path / "unused")
    assert path.read_bytes() == unchanged and not (tmp_path / "unused").exists()


def test_cleanup_detects_changed_file_before_replacement(tmp_path):
    path = tmp_path / "4.txt"
    original = block("0000", "0011") + block("0000", "0110")
    path.write_text(original)
    plan = plan_cleanup(path)
    changed = original + block("0011", "0101")
    path.write_text(changed)
    with pytest.raises(RuntimeError, match="changed"):
        apply_cleanup([plan], tmp_path / "backup")
    assert path.read_text() == changed


def test_cleanup_rejects_invalid_or_malformed_record_without_modification(tmp_path):
    path = tmp_path / "4.txt"
    path.write_text(block("0000", "0011") + "L=4\na=0000\nb=0000\n")
    original = path.read_bytes()
    with pytest.raises(ValueError): plan_cleanup(path)
    assert path.read_bytes() == original


def test_cleanup_rejects_metadata_mismatch(tmp_path):
    path = tmp_path / "4.txt"
    path.write_text(block("0000", "0011").replace("nonzero PACS=4", "nonzero PACS=-4"))
    with pytest.raises(ValueError, match="metadata"): plan_cleanup(path)
