"""Complete one L86 fixed-content sparse-support subfamily.

For fixed coarse negative count n_A, |U|=sigma_A+2*n_A and W=1-V
has the same size. Their complete mod-2 difference signatures must XOR
to all independent bits except the target. If these supports are dense,
complement both: the XOR condition remains identical. Retain every
bracelet preimage and solve both independent exact sign problems.
Independent binary reversal/even rotation permits these representatives.
Original k=2 or k=41 both fold to shift 2; their lift targets differ.
This covers the chosen content/target only, not all L86 PQCPs.
"""
import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.checkpoint import atomic_write_json
from solver.pqcp_support_solver import complete_pqcp_supports
from solver.search_runner import append_verified_solution_if_new
from solver.sparse_supports import sparse_bracelets
from solver.target_profiles import TargetContentProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eta', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--negative-a', type=int, default=0)
    parser.add_argument('--stop-after-new', type=int, default=0)
    parser.add_argument('--odd-target', action='store_true', help='original k=41 folds to independent shift 2')
    parser.add_argument('--coarse-only', action='store_true', help='exhaust exact signed compression only; never write binary solutions')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not 0 <= args.negative_a <= 16 or args.stop_after_new < 0:
        parser.error('negative-a in 0..16 and nonnegative stopping count required')
    started = perf_counter()
    weight = (6 if args.eta == 1 else 5)+2*args.negative_a
    sparse_weight = min(weight, 43-weight)
    generated = sparse_bracelets(43, sparse_weight, capacity=min(comb(43, sparse_weight)//43, 2**23))
    if not generated['complete']:
        raise RuntimeError('bracelet capacity truncated; exhaustive subfamily claim invalid')
    index = {}
    for mask, signature in generated['records']:
        index.setdefault(signature, []).append(mask)  # ALL literal bracelet preimages.
    # V=1+W in F2, and P_(1+W)=1+P_W at nonzero shifts for odd d.
    # P_U+P_V=delta_k therefore means sig(U)+sig(W)=all_ones+delta_k.
    target_signature = ((1 << 21)-1) ^ (1 << (2-1))
    pair_count = sum(len(index.get(signature ^ target_signature, ())) for _, signature in generated['records'])
    print('bracelets', generated['total'], 'matching ordered support pairs', pair_count, flush=True)
    target = (TargetContentProfile(86, 2, 1, 17, 20, 17, 23) if args.eta == 1 else
              TargetContentProfile(86, 2, -1, 17, 21, 17, 22))
    if args.odd_target:
        target = (TargetContentProfile(86, 41, 1, 16, 21, 18, 22) if args.eta == 1 else
                  TargetContentProfile(86, 41, -1, 16, 22, 18, 21))
    na, nb = args.negative_a, 17-args.negative_a
    a, b = tuple([-1]*na+[0]*(43-weight)+[1]*(weight-na)), tuple([-1]*nb+[0]*weight+[1]*(43-weight-nb))
    rows = []
    statuses, exclusions = Counter(), Counter()
    tested = verified = new = 0
    solve_seconds = 0.0
    first_verified_elapsed = None
    coarse_verified = 0
    stopped = False
    for uu, signature in generated['records']:
        for ww in index.get(signature ^ target_signature, ()):
            u, w = (((1 << 43)-1) ^ uu, ((1 << 43)-1) ^ ww) if weight > 21 else (uu, ww)
            masks = (u, ((1 << 43)-1) ^ w)
            # Reproduce proofs without recorded content exclusions. The
            # coarse-only path bypasses both runtime content pruners entirely.
            if args.coarse_only:
                from solver.ternary_system_native import prepare_native_system, run_native_system
                from solver.structured_energy import fold_profile, target_profile
                from solver.ternary_support_search import _audit_sign_words
                tick = perf_counter()
                wanted = tuple(v//4 for v in fold_profile(target_profile(86, target.k, target.eta), 2))
                plan = prepare_native_system(masks, 43, wanted, (na, nb))
                sign = run_native_system(plan, 2**23)
                if sign['words'] is not None:
                    _audit_sign_words(sign['words'], a, b, target, wanted, masks)
                    coarse_verified += 1
                out = dict(status=sign['status'], words=sign['words'], verified=False,
                           coarse_verified=sign['words'] is not None, attempts=dict(coarse=sign),
                           excluded_by='coarse' if sign['status'] == 'UNSAT_FIXED_SUPPORT' else None,
                           elapsed=perf_counter()-tick)
            else:
                out = complete_pqcp_supports(a, b, target, masks, binary_content_pruning=False)
            tested += 1
            statuses[out['status']] += 1
            exclusions[out['excluded_by'] if out['excluded_by'] is not None else 'none'] += 1
            solve_seconds += out['elapsed']
            row = dict(u=u, w=w, result=out)
            if out['verified']:
                verified += 1
                if first_verified_elapsed is None:
                    first_verified_elapsed = perf_counter()-started
                row['new_isolated_solution'] = append_verified_solution_if_new(86, *out['words'], args.output)
                new += bool(row['new_isolated_solution'])
            # Large complete searches retain witnesses and unresolved cases,
            # plus reproducible aggregate coverage instead of millions of
            # redundant UNSAT dictionaries.
            coarse_sat = out['attempts'].get('coarse', {}).get('status') == 'SAT_COMPRESSED'
            if sparse_weight <= 6 or out['verified'] or out['status'] == 'UNKNOWN' or coarse_sat:
                rows.append(row)
            if tested % (64 if sparse_weight <= 6 else 4096) == 0 or out['verified'] or tested == pair_count:
                progress = dict(tested=tested, total=pair_count, verified=verified, coarse_verified=coarse_verified, new=new,
                                statuses=dict(statuses), elapsed=perf_counter()-started)
                atomic_write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
            if args.stop_after_new and new >= args.stop_after_new:
                stopped = True
                break
        if stopped:
            break
    summary = dict(target=target.__dict__, restricted_family='n_A={}, k={}, eta={:+d}'.format(na, target.k, args.eta),
        completion_scope='compressed' if args.coarse_only else 'binary',
        bracelet_generation={k:v for k,v in generated.items() if k != 'records'},
        distinct_parity_signatures=len(index), matched_pairs=pair_count, tested=tested,
        statuses=dict(statuses), excluded_by=dict(exclusions), verified=verified, coarse_verified=coarse_verified, new=new,
        seconds=solve_seconds, total_elapsed=perf_counter()-started,
        first_verified_elapsed=first_verified_elapsed,
        complete=not stopped and tested == pair_count and statuses['UNKNOWN'] == 0)
    atomic_write_json(args.output/'report.json', dict(summary=summary, rows=rows))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
