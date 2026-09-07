"""Cross-check optimized C arithmetic with independent exact Python PACF."""

import ctypes
import itertools
import os
from pathlib import Path
import random
import subprocess

import pytest

from solver.correlation import full_correlation_profile
from solver.objective import pqcp_objective
from solver.c_backend import ensure_compressed_fkm_seed_bank


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    library = tmp_path_factory.mktemp("c-kernel") / "probe.so"
    source = Path(__file__).with_name("c_kernel_probe.c")
    subprocess.run([
        "cc", "-O3", "-std=c11", "-shared", "-fPIC", "-pthread",
        str(source), "-lm", "-o", str(library),
    ], check=True, capture_output=True)
    dll = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_byte)
    dll.probe_delta.argtypes = [ptr] + [ctypes.c_int] * 4
    dll.probe_energy.argtypes = [ptr, ptr] + [ctypes.c_int] * 6
    dll.probe_energy.restype = ctypes.c_longlong
    dll.probe_remainder.argtypes = [ctypes.c_uint64, ctypes.c_int]
    dll.probe_remainder.restype = ctypes.c_uint64
    dll.probe_pool_pairs.argtypes = [ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    dll.probe_pool_trace.argtypes = [ptr, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int)]
    dll.probe_quarter.argtypes = [ctypes.c_int] * 7
    return dll


@pytest.mark.parametrize("length", [4, 6, 8, 10, 16, 44, 46, 58, 68, 86, 90, 94])
def test_swap_energy_matches_full_python_recomputation(kernel, length):
    rng = random.Random(60500 + length)
    for _ in range(24):
        a = [rng.randrange(2) for _ in range(length)]
        b = [rng.randrange(2) for _ in range(length)]
        p, q = rng.sample(range(length), 2)
        k = rng.randrange(1, length // 2)
        sign = rng.choice((-1, 1))
        ca = (ctypes.c_byte * length)(*(1 - 2 * x for x in a))
        cb = (ctypes.c_byte * length)(*(1 - 2 * x for x in b))
        before = full_correlation_profile(a, b)
        a[p] ^= 1
        a[q] ^= 1
        after = full_correlation_profile(a, b)
        for u in range(1, length // 2 + 1):
            assert kernel.probe_delta(ca, length, p, q, u) == after[u] - before[u]
        residual = after[:]
        residual[0] -= 2 * length
        residual[k] -= 4 * sign
        residual[length - k] -= 4 * sign
        expected = sum(x * x for x in residual[1:length // 2 + 1])
        for factor in (2, 4):
            if length % factor:
                continue
            d = length // factor
            expected += factor * sum(
                sum(residual[r::d]) ** 2 for r in range(d)
            )
        assert kernel.probe_energy(ca, cb, length, p, q, k, sign, 0) == expected
        assert kernel.probe_energy(ca, cb, length, p, q, k, sign, 1) == expected


@pytest.mark.parametrize("length", [4, 44, 46])
@pytest.mark.parametrize("quench,move_pool", [(False, False), (True, False),
                                             (False, True), (True, True)])
def test_reference_and_fast_search_have_identical_restart_trajectory(
    tmp_path, length, quench, move_pool
):
    """A fixed-work, one-thread run must produce the exact same sequence pairs."""
    bank = ensure_compressed_fkm_seed_bank(length, 123, tmp_path, seeds_per_content=2)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PQCP_")}
    environment.update(PQCP_SEED="123", PQCP_THREADS="1", PQCP_CANDIDATES="2",
                       PQCP_FKM_SEEDS=str(bank), PQCP_MAX_RESTARTS="8",
                       PQCP_EMIT_ELITES="1")
    if quench:
        environment["PQCP_QUENCH"] = "1"
    if move_pool:
        environment["PQCP_MOVE_POOL"] = "1"
    outputs = []
    source = Path(__file__).resolve().parents[1] / "csrc/pqcp_search.c"
    for reference in (True, False):
        executable = tmp_path / ("reference" if reference else "optimized")
        flags = ["-DPQCP_REFERENCE_KERNEL"] if reference else []
        subprocess.run(["cc", "-O3", "-std=c11", "-pthread", *flags,
                        str(source), "-lm", "-o", str(executable)],
                       check=True, capture_output=True)
        output = subprocess.run([str(executable), str(length)], env=environment,
                                check=True, capture_output=True, text=True,
                                timeout=30).stdout
        rows = output.splitlines()
        elites = [line for line in rows if line.startswith("PQCP_ELITE ")]
        candidates = [line for line in rows if line.startswith("PQCP_CANDIDATE ")]
        stats = [line for line in rows if line.startswith("搜尋統計：")][-1]
        assert "restart=8," in stats
        assert elites
        for line in elites:
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            profile = full_correlation_profile(fields["a"], fields["b"])
            assert int(fields["score"]) == pqcp_objective(profile)
            k, sign = int(fields["k"]), int(fields["sign"])
            profile[0] -= 2 * length
            profile[k] -= 4 * sign
            profile[length - k] -= 4 * sign
            expected = sum(x * x for x in profile[1:length // 2 + 1])
            for m in (2, 4):
                if length % m == 0:
                    d = length // m
                    expected += m * sum(sum(profile[r::d]) ** 2 for r in range(d))
            assert int(fields["energy"]) == expected
        outputs.append((elites, candidates, stats))
    assert outputs[0] == outputs[1]


def test_exact_remainder_at_uint64_boundaries_and_random_inputs(kernel):
    rng = random.Random(714)
    for length in [4, 6, 8, 44, 46, 58, 68, 86, 90, 94, 128, 1000]:
        values = [0, 1, length - 1, length, length + 1, 2 ** 64 - 1,
                  2 ** 64 - 2, 2 ** 63]
        values += [rng.getrandbits(64) for _ in range(1000)]
        for value in values:
            assert kernel.probe_remainder(value, length) == value % length


def test_pool_indexes_each_legal_swap_exactly_once(kernel):
    for length in (4, 6, 8):
        for signs in itertools.product((-1, 1), repeat=length):
            expected = {(i, j) for i in range(length) for j in range(i + 1, length)
                        if i % 2 == j % 2 and signs[i] != signs[j]}
            sequence = (ctypes.c_byte * length)(*signs)
            pairs = (ctypes.c_int * (length * length))()
            count = kernel.probe_pool_pairs(sequence, length, pairs)
            actual = [tuple(sorted((pairs[2 * i], pairs[2 * i + 1])))
                      for i in range(count)]
            assert len(actual) == len(set(actual))
            assert set(actual) == expected


@pytest.mark.parametrize("length", [4, 8, 44, 46, 68, 94])
def test_pool_stays_consistent_after_repeated_swaps(kernel, length):
    rng = random.Random(length)
    original = [rng.choice((-1, 1)) for _ in range(length)]
    sequence = (ctypes.c_byte * length)(*original)
    pairs = (ctypes.c_int * 400)()
    count = kernel.probe_pool_trace(sequence, length, 200, pairs)
    assert count >= 0
    replay = original[:]
    for step in range(count):
        p, q = pairs[2 * step], pairs[2 * step + 1]
        assert p % 2 == q % 2 and replay[p] != replay[q]
        replay[p], replay[q] = replay[q], replay[p]
        assert sum(replay[::2]) == sum(original[::2])
        assert sum(replay[1::2]) == sum(original[1::2])
    assert list(sequence) == replay


@pytest.mark.parametrize("length", [4, 8, 12])
def test_quarter_pruning_matches_exhaustive_binary_sequence_spectra(kernel, length):
    # Build attainable spectra from actual full sequences, not the pruning
    # formula. The Cartesian product of these grouped spectra covers all
    # sequence pairs without an expensive 2^(2L) pair enumeration.
    attainable = {}
    for bits in itertools.product((0, 1), repeat=length):
        signs = [1 - 2 * bit for bit in bits]
        real = sum(signs[::4]) - sum(signs[2::4])
        imag = sum(signs[1::4]) - sum(signs[3::4])
        content = (sum(bits[::2]), sum(bits[1::2]))
        attainable.setdefault(content, set()).add(real * real + imag * imag)
    for (ae, ao), left in attainable.items():
        for (be, bo), right in attainable.items():
            pair_spectra = {a + b for a in left for b in right}
            for k in range(1, length // 2):
                for sign in (-1, 1):
                    cosine = (1, 0, -1, 0)[k % 4]
                    target = 2 * length + 8 * sign * cosine
                    assert bool(kernel.probe_quarter(length, k, sign, ae, ao, be, bo)) == (
                        target in pair_spectra)


def test_quarter_pruning_rejects_impossible_l44_k2_parity_content(kernel):
    assert not kernel.probe_quarter(44, 2, -1, 9, 11, 7, 11)
    assert kernel.probe_quarter(44, 2, -1, 8, 12, 8, 10)
    assert kernel.probe_quarter(44, 4, -1, 9, 11, 7, 11)
    assert kernel.probe_quarter(46, 2, 1, 9, 11, 7, 11)
