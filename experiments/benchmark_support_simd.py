"""Paired exact CRT-kernel comparison against a preserved scalar library."""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.cyclic_support import parity_support_family
from solver.cyclic_support_native import _library
from solver.ternary_support_search import support_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--baseline-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if min(args.repeats, args.iterations) < 1 or args.output.exists():
        parser.error('positive repeat/iteration counts and a new output file required')
    current = _library()
    legacy = ctypes.CDLL(str(args.baseline_library.resolve()))
    legacy.pqcp_nearest_supports.argtypes = current.pqcp_nearest_supports.argtypes
    legacy.pqcp_nearest_supports.restype = ctypes.c_int
    array = ctypes.c_uint64
    prepared = []
    for row in json.loads(args.input.read_text()):
        words = row['result']['words']
        for side in (0, 1):
            family = parity_support_family(support_mask(words[side]), 43, row['reference']['target']['k'])
            groups = list(family.choices)+[(0,)]*(3-len(family.choices))
            arrays = [(array*len(group))(*group) for group in groups]
            prepared.append([arrays[0], len(groups[0]), arrays[1], len(groups[1]),
                arrays[2], len(groups[2]), family.base, support_mask(words[1-side]),
                sum(v != 0 for v in words[1-side])])

    def invoke(lib, case):
        output = (array*8)()
        stats = (array*3)()
        distance = lib.pqcp_nearest_supports(*case, output, 8, stats)
        return distance, tuple(stats), tuple(output[:stats[2]])

    for case in prepared:
        if invoke(legacy, case) != invoke(current, case):
            raise RuntimeError('scalar and SIMD disagree on exact masks or full counts')
    samples = []
    for repeat in range(args.repeats):
        sample = {}
        for name, lib in ((('scalar', legacy), ('simd', current)) if repeat % 2 == 0 else
                          (('simd', current), ('scalar', legacy))):
            started = perf_counter()
            for _ in range(args.iterations):
                for case in prepared:
                    invoke(lib, case)
            sample[name] = perf_counter()-started
        samples.append(sample)
        print('repeat', repeat+1, json.dumps(sample), flush=True)
    scalar = statistics.median(s['scalar'] for s in samples)
    simd = statistics.median(s['simd'] for s in samples)
    summary = dict(cases=len(prepared), iterations=args.iterations, repeats=args.repeats,
        scalar_seconds=scalar, simd_seconds=simd, exact_outputs_equal=True,
        seconds_reduction_percent=100*(1-simd/scalar),
        throughput_increase_percent=100*(scalar/simd-1))
    args.output.write_text(json.dumps(dict(summary=summary, samples=samples), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
