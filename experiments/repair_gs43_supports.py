"""Apply existing exact cyclic full completion to GS-derived nearby seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.benchmark_l86_discovery import discovery_key
from solver.checkpoint import atomic_write_json
from solver.correlation import full_correlation_profile
from solver.cyclic_support_search import repair_cyclic_supports
from solver.search_runner import append_verified_solution_if_new
from solver.solution_equivalence import pqcp_equivalence_key
from solver.target_profiles import TargetContentProfile, pair_content
from solver.torch_joint_search import binary_joint_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/L86_GS_interleave_20261007/report.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-supports', type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.max_supports <= 4096:
        parser.error('bounded support count 1..4096 required')
    args.output.mkdir(parents=True, exist_ok=False)
    saved, seen, jobs = json.loads(args.input.read_text()), set(), []
    for seed in saved['seeds']:
        words = tuple(map(tuple, seed['words']))
        key = pqcp_equivalence_key(*words)
        if key in seen:
            continue
        seen.add(key)
        profile = full_correlation_profile(*words)
        for k in range(1, 43, 2):
            if profile[k] != 4:
                continue
            target = TargetContentProfile(86, k, 1, *pair_content(*words))
            c, _ = binary_joint_rows(words, target)
            jobs.append((c, target))
    tick, rows, keys, first_hit = perf_counter(), [], set(), None
    for index, (c, target) in enumerate(jobs):
        out = repair_cyclic_supports(*c, target, max_supports=args.max_supports, full_completion=True)
        if out['full_words'] is not None:
            key = discovery_key(*out['full_words'])
            if key not in keys and first_hit is None:
                first_hit = perf_counter()-tick
            keys.add(key)
            append_verified_solution_if_new(86, *out['full_words'], args.output)
        rows.append(dict(k=target.k, reference=c, result=out))
        report = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            max_supports=args.max_supports, jobs=len(jobs), rows=rows, complete=len(rows)==len(jobs),
            independently_verified_new_classes=len(keys), first_verified_elapsed=first_hit,
            elapsed=perf_counter()-tick, scope='Bounded fixed-axis support repair from GS seeds; misses do not exclude other supports.')
        atomic_write_json(args.output/'report.json', report)
        print(json.dumps(dict(job=index+1, total=len(jobs), k=target.k, status=out['status'],
            supports=len(out['attempts']), elapsed=out['elapsed'], new_classes=len(keys))), flush=True)


if __name__ == '__main__':
    main()
