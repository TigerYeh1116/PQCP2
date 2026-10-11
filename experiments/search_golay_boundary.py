"""Exact L86 boundary search from a fully specified generated Golay-80 family.

Inspired by the 2026 ICC PQCP paper, but imposing this project's strict target.
All prefix-prefix PACF sums cancel; mixed terms are linear, tail-tail terms
quadratic. Complements/reversals/row exchange quotient prefix families only;
all 12 tail bits and every strict target shift/sign remain in each query.
"""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import z3
from solver.verifier import verify_pqcp
from solver.search_runner import append_verified_solution_if_new

PAPER = 'https://repository.essex.ac.uk/42886/1/Final%20Version.pdf'


def variants(row):
    return {row, row[::-1], tuple(-v for v in row), tuple(-v for v in row[::-1])}


def canonical(pair):
    return tuple(sorted(min(variants(row)) for row in pair))


def audit_golay(pair):
    n = len(pair[0])
    assert len(pair[1]) == n
    assert all(v in (-1, 1) for row in pair for v in row)
    assert all(sum(row[i]*row[i+u] for row in pair for i in range(n-u)) == 0
               for u in range(1, n))


def generate_prefixes():
    p = tuple(1 if c == '+' else -1 for c in '+--+-+---+')
    g = tuple(1 if c == '+' else -1 for c in '+------++-')
    audit_golay((p, g))
    families, counts = {canonical((p, g))}, []
    for _ in range(3):
        output = set()
        for a, b in families:
            for left, right in ((a, b), (b, a)):
                for x, y in product(variants(left), variants(right)):
                    negative_y = tuple(-v for v in y)
                    output.add(canonical((x+y, x+negative_y)))
                    output.add(canonical((tuple(v for xy in zip(x, y) for v in xy),
                                          tuple(v for xy in zip(x, negative_y) for v in xy))))
        families = output
        for pair in families:
            audit_golay(pair)
        counts.append((len(next(iter(families))[0]), len(families)))
    return sorted(families), counts


def boundary_coefficients(pair, tail, u):
    """Mixed linear and directed tail-tail quadratic terms; prefix part cancels."""
    prefix_length = len(pair[0])
    length = prefix_length+tail
    if tail < 1 or len(pair[1]) != prefix_length or not 1 <= u < length:
        raise ValueError('equal prefixes, nonempty tail, and nonzero shift required')
    linear, edges = [], []
    for side, prefix in enumerate(pair):
        for j in range(tail):
            pos = prefix_length+j
            linear.append(sum(prefix[z] for z in ((pos+u) % length, (pos-u) % length) if z < prefix_length))
            other = (pos+u) % length
            if other >= prefix_length:
                edges.append((side*tail+j, side*tail+other-prefix_length))
    return tuple(linear), tuple(edges)


def solve_prefix(pair, timeout_ms=30000):
    audit_golay(pair)
    tail, length = 6, 86
    if len(pair[0]) != length-tail:
        raise ValueError('Golay-80 prefixes required')
    bits = [z3.Bool('tail_'+str(i)) for i in range(2*tail)]
    signs = [z3.If(v, -1, 1) for v in bits]
    k, eta = z3.Int('k'), z3.Bool('eta')
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(k >= 1, k < length//2)
    for u in range(1, length//2+1):
        linear, edges = boundary_coefficients(pair, tail, u)
        terms = [coefficient*sign for coefficient, sign in zip(linear, signs) if coefficient]
        terms.extend(z3.If(bits[a] == bits[b], 1, -1) for a, b in edges)
        solver.add(z3.Sum(terms) == z3.If(k == u, 4*z3.If(eta, 1, -1), 0))
    started = perf_counter()
    status = solver.check()
    result = dict(status=str(status), elapsed=perf_counter()-started,
        reason=solver.reason_unknown() if status == z3.unknown else None, words=None)
    if status == z3.sat:
        model = solver.model()
        words = tuple(tuple((1-v)//2 for v in prefix)+tuple(
            int(z3.is_true(model.eval(v, model_completion=True))) for v in bits[side*tail:(side+1)*tail])
            for side, prefix in enumerate(pair))
        check = verify_pqcp(*words)
        if not check.is_valid:
            raise RuntimeError('Strict L86 boundary witness failed independent full verifier')
        result.update(words=words, k=model.eval(k).as_long(),
                      eta=1 if z3.is_true(model.eval(eta, model_completion=True)) else -1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-ms', type=int, default=30000)
    args = parser.parse_args()
    if args.timeout_ms <= 0:
        raise ValueError('positive query budget required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    prefixes, generation = generate_prefixes()
    rows, statuses, new = [], Counter(), 0
    for i, pair in enumerate(prefixes):
        result = solve_prefix(pair, args.timeout_ms)
        statuses[result['status']] += 1
        if result['words'] is not None:
            new += append_verified_solution_if_new(86, *result['words'], args.output)
        rows.append(dict(prefix_sha256=hashlib.sha256(bytes((1-v)//2 for row in pair for v in row)).hexdigest(),
                         prefix=pair, result=result))
        if (i+1) % 16 == 0:
            print(json.dumps(dict(tested=i+1, total=len(prefixes), statuses=dict(statuses))), flush=True)
    report = dict(generation=generation, tested=len(rows), statuses=dict(statuses),
        complete=not statuses['unknown'], new_verified_in_isolated_archive=new,
        elapsed=perf_counter()-started, rows=rows, literature=PAPER,
        scope='Only this generated Golay-80 family with arbitrary six appended bits per row; not all PQCPs.',
        speed_claim='No new-solution speed claim from UNSAT or generation counts.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
