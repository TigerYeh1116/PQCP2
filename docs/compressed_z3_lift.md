# Exact factor-2 Z3 lift completion

## Scope and integration

With `main.py --z3`, the production PyTorch hybrid and C backends now feed
their existing elite callbacks into a bounded compressed-pattern pool.
The same bridge is used by the Adam/resume entry point. No FKM, loss,
correlation, SA move/acceptance, or verifier definition changes are needed.
The separate legacy Python runner is unchanged.

The pool hashes the existing exact factor-2 PACF signatures by parity
content. A new side is matched against archived partner sides. This reuses
existing FKM/PyTorch/C candidates; it does **not** yet exhaustively enumerate
all compressed ternary patterns. Retention limits are heuristics, not hard
mathematical pruning or completeness guarantees.

An exact compressed match queues a lift regardless of score, global-best
improvement or the original Hamming lower bound. The original radius
completion is still available for eligible improving centers. A single
background worker runs both operations while GPU formation and C annealing
continue. Different compressed jobs are not debounced away as obsolete
scores. The new queue admission cap is 16; each content pool retains at most
64 patterns and scheduled-job history retains 512 keys.

Admission now defaults to [compressed-job symmetry keys](compressed_job_equivalence.md),
transforming the full target position and parity content together with the
patterns. Stored representatives stay in their original coordinates. Use
`--no-z3-compressed-dedup` to restore exact-identity-only admission.

## Mathematical model

For bipolar signs and L=2d:

```
c[i] = x[i] + x[i+d]
c[i]=+2 -> Project bits (0,0)
c[i]=-2 -> Project bits (1,1)
c[i]= 0 -> (v,1-v), one Boolean orientation v
```

Positions are `i` and `i+d`, never adjacent expansion. Before constructing
Z3, both compressed PACFs must sum exactly to `fold_2(T)`, where
`T(0)=2L`, `T(k)=T(L-k)=4*eta`, and all other entries are zero.
The compressed diagonal is the folded full target, **not** twice the
compressed length.

Let z_A,z_B count compressed zeros. Since `rho_x(d)=L-4*z_x` and T(d)=0,
the exact match forces `z_A+z_B=d`. Thus the primary orientation variable
count is 22 at L=44, 23 at L=46, and 34 at L=68. These counts do not include
Z3's internal encoding auxiliaries and are not a runtime guarantee.

For each independent shift 1..d-1 the model counts all cyclic mismatches
across both words and enforces `m(u)=L-T(u)/2`, equivalent to the exact
correlation `S(u)=2L-2*m(u)`. S(0), S(d) and periodic symmetry are guaranteed
by construction. There is no center or Hamming-distance bound.

Optional exact parity content is enforced on all four groups. When d is
even it is already fixed by the compressed pattern; when d is odd it is
essential because zero orientations can change alternating sums.

SAT must pass the unchanged independent verifier AND this selected target,
compression and content. UNSAT excludes only this fixed compression/content
job. UNKNOWN excludes nothing; solver reason and timings are saved.

## Usage and rollback

The existing interactive Z3 question is enough: choosing yes enables this
path by default. CLI example:

```
python main.py --L 68 --z3
```

Portable device selection and the existing loss options are unchanged.
To restore the prior completion path:

```
python main.py --L 68 --z3 --no-z3-compressed-lift
```

`--z3-timeout` applies to each lift's Z3 check as well as the original
completion. Construction time is recorded separately. Ctrl+C preserves an
in-flight result and waits for that task to finish (potentially up to the
configured solver timeout); pending jobs are not drained, as in the previous
bridge. The new pool is not checkpointed and will rebuild after resume.

## Persistence and validation

For lengths divisible by four, the background bridge now also offers exact
factor-4 pairs to the [multi-level lift](multilevel_z3_lift.md), enabled by
default with Z3. This factor-2 path remains available independently.

Every attempted lift saves an atomic JSON snapshot under
`results/compressed_lift/L{L}_seed{seed}_{job_hash}.json`, including the fixed
patterns, content, source words, status, reason, counts and construction/solve
timings. SAT includes full A/B, exact profile, nonzero shifts/values and
objective components. The candidate is independently re-audited before the
existing thread-safe writer appends a new pair to the established `L.txt`
format, including symmetry-group deduplication (independent row rotation,
reversal and complementation, A/B exchange, and common unit decimation).
New solutions also get a
non-overwriting snapshot in `results/verified/`.

Tests compare **all exact-matched compressed pattern pairs** at L=4,6,8
against unrestricted full binary enumeration, including matches with no
binary lift. Additional tests check odd-d content, UNKNOWN, verifier failure,
antipodal placement, persistence, guards, rollback, and queue semantics.
No known large PQCP is used to initialize this search path. Finding a valid
compressed match does not guarantee a lift or improved solution throughput.
