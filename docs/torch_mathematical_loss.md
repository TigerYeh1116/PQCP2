# PyTorch mathematical loss experiments

## Scope

These losses guide only the relaxed PyTorch/MPS population. They do not
change the Project 2 target, exact integer objective, C completion, Z3, result
writer, or independent verifier. No known PQCP is used for initialization or
optimization. Existing solutions are used only once as a correctness audit:
all new necessary terms must vanish on them.

The two supplied PACP slide decks motivated GPU populations, quantization,
rebirth, multi-factor folds, and two-stage fitness. Their PACP target formulas
were not copied. Every formula below is re-derived for this project's target

```
T(0)=2L, T(k)=T(L-k)=4*eta, T(u)=0 otherwise.
```

## New mathematical terms

### Individual PSD cap

Wiener--Khinchin gives

```
PSD_A(f) + PSD_B(f) = DFT(T)(f)
                    = 2L + 8*eta*cos(2*pi*f*k/L).
```

Both individual power spectra are nonnegative, so neither may exceed the
right-hand side. `psd_cap` applies a squared hinge to violations. This is a
necessary condition, not hard pruning. It is often inactive early in search.

### Every valid compression factor and lift alphabet

For `L=m*d`, genuine compression is

```
c[r] = sum_(q=0..m-1) x[r+q*d].
```

Its PACF is the corresponding fold of the original PACF. The previous loss
used only factors 2 and 4. `divisor_lift` adds residual folds for every other
proper factor and checks the relaxed compressed entries against the necessary
binary lift alphabet `{-m,-m+2,...,m}`. Factors are averaged so the divisor
count does not itself create a length-dependent coefficient.

### Fixed-weight individual PACF lattice

For a binary sign sequence of weight `w`, let `q` be the intersection size of
its support and its shift. Then

```
rho_x(u) = L - 4*(w-q) = L - 4h,
0 <= h <= min(w,L-w), h integer.
```

`lattice` projects each relaxed *individual* PACF value to this finite set.
The pair sum alone can otherwise look good while A and B separately have
values that no fixed-weight binary sequence can realize. The nearest lattice
choice is detached; the loss is zero for every fixed-weight binary sequence.

### Exact Bernoulli expected-error correction

Interpret a relaxed sign `m_i` as the mean of an independent random sign
`X_i`. Existing squared PACF residual is only the squared mean error. The
mathematically complete expected error is

```
E[(C(u)-T(u))^2] = (E[C(u)]-T(u))^2 + Var(C(u)).
```

For non-half shifts, with `Y_i=X_i*X_(i+u)`, only adjacent cycle edges have
nonzero covariance:

```
Var rho(u) = sum_i (1-m_i^2*m_(i+u)^2)
           + 2 sum_i m_i*m_(i+2u)*(1-m_(i+u)^2).
```

At `u=L/2`, each edge is counted twice and

```
Var rho(L/2) = 4 sum_(i<L/2) (1-m_i^2*m_(i+L/2)^2).
```

The implementation sums the covariance across shifts using a cyclic
self-convolution reconstructed from the existing real DFT matrices. This
avoids a large batch-by-shift-by-position autograd tape. Exhaustive probability
enumeration at L=4 and L=6 independently verifies the formula.

`combined` sums all of the above in the same four-unit correlation scale.
`lattice_bootstrap` uses the lattice term during the first of the four existing
cosine learning-rate cycles, then returns to the original target loss. No
fractional heuristic coefficient was introduced.

## Correctness audit

- 432 independently verified pairs in `44.txt` were inspected only after the
  formulas were implemented. Maximum double-precision residuals were about
  `2.8e-28` (PSD cap), `8.8e-28` (new divisor folds), `0` (liftability and
  Bernoulli variance), and `2.2e-13` (DFT roundoff in PACF lattice).
- Every random binary pair in tests has zero liftability, PACF-lattice, and
  Bernoulli-variance penalty.
- The `none` path remains bit-identical to the previous loss.
- Checkpoint/resume remains deterministic for enabled mathematical losses.

## Measured results

Loss-only pilots used actual MPS, identical FKM initialization, and exact
post-quantization scores. They showed that full-time `combined` and
`variance` reduced MPS epoch throughput and did not improve the median exact
candidate score. This was treated as diagnostic, not as grounds to delete the
new mathematics.

The more promising lattice term was then tested end-to-end with real MPS and
C completion, 120 seconds per arm, using paired new seeds and a frozen result
inventory only for deduplication:

| L | seeds | method | new pairs | total seconds | seconds/new |
|---|---:|---|---:|---:|---:|
| 44 | 5 | previous loss | 8 | 607.88 | 75.99 |
| 44 | 5 | lattice | 6 | 607.93 | 101.32 |
| 46 | 3 | previous loss | 0 | 364.03 | not reached |
| 46 | 3 | lattice | 0 | 364.18 | not reached |

The first three L=44 seeds initially favored lattice 6:5, but two untouched
holdout seeds favored the previous loss 3:0. This is why the smaller favorable
result was not promoted. L=46 is inconclusive at this budget, not a proof of
failure or nonexistence.

Of the eight baseline discoveries, seven came from C completion and one from
MPS polish; all six lattice discoveries came from C completion. Several C
discoveries occurred before the first MPS bank handoff, so they cannot be
credited to the new loss. Thread scheduling also changes C throughput between
arms. The end-to-end totals are therefore the correct user-facing yield
measurement, but not evidence that lattice caused an improvement.

A separate three-seed, 20-second loss-only test of `lattice_bootstrap` also did
not improve exact projected scores (median 8 vs 7; minima 8 vs 4). It remains
available because that test is short, but is not the default.

Artifacts are under:

- `results/torch_math_loss_20260911_*`
- `results/torch_yield_20260911_math_lattice`
- `results/torch_yield_20260911_math_lattice_holdout`
- `results/torch_yield_20260911_math_lattice_L46`

## Decision

The original loss remains the production default because the requested
improvement is not established. All structural modes are selectable through
`--torch-mathematical-loss` (`psd_cap`, `divisor_lift`, `lattice`, `variance`,
`combined`, or `lattice_bootstrap`) and can be removed from a run with `none`.
This preserves a clean rollback path and prevents a mathematically attractive
but empirically slower loss from silently reducing new-solution throughput.

Primary mathematical references:

- D. Z. Djokovic and I. S. Kotsireas, *Compression of Periodic Complementary
  Sequences and Applications*, arXiv:1302.0571.
- B. Boskovic and J. Brest, *Two-phase Optimization of Binary Sequences with
  Low Peak Sidelobe Level Value*, arXiv:2107.09801. The two-fitness concept is
  motivational only; its PSL objective is not this project's PQCP objective.
