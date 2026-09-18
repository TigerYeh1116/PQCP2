# PQCP continuous-loss ablation

## Scope and source definitions

Project 2 question 3 (`文獻/06_PACP.pdf`, page 29) and the unchanged
`problem.py`, `correlation.py`, `verifier.py` define the problem. For signs
`x=(-1)^A`, `y=(-1)^B`, use

```
C(u) = sum_i x_i x_(i+u mod L) + y_i y_(i+u mod L)
T(0) = 2L; T(k) = T(L-k) = 4 eta; other T(u) = 0
0 < k < L/2; eta in {-1,+1}; e = C-T
```

Two actual sidelobes are nonzero, but they are one reflection orbit. The
independent residual sum runs through `1..L/2` once. The full target is used
when folding, including the origin and any colliding target shifts.

The supplied odd/even PACP slides motivate continuous optimization, explicit
sum constraints, folding, and independent quantized evaluation. Their odd
`|C(u)|=2` target and even half-shift-only target are **not** PQCP targets and
are not imported. In particular, here `T(L/2)=0`.

[Song, Babu & Palomar, *Sequence Design to Minimize the Weighted Integrated
and Peak Sidelobe Levels* (2015)](https://arxiv.org/abs/1506.04234) studies
weighted sidelobe minimization and FFT-based optimization. It supports the
general idea of weighted residual objectives, not the claim that their
aperiodic/unimodular algorithms solve this binary periodic PQCP problem.
The projection and folding normalization below are direct derivations for
this project's loss, not formulas attributed to that paper.

## Existing exact targets and content

The existing target profiles already satisfy the necessary Fourier identities

```
(sum x)^2 + (sum y)^2 = 2L + 8 eta
Alt(x)^2 + Alt(y)^2 = 2L + 8 eta (-1)^k
```

Thus every lane specifies the number of negative signs in each of A-even,
A-odd, B-even and B-odd. These are necessary, not sufficient, conditions.
Relaxed signs need not obey them exactly; rank projection always does.

## Fold normalization

Let `(F_m e)[r] = sum_q e[r+qL/m]` and `d=L/m`. The disjoint rows of `F_m`
have m ones, so `F_m F_m^T=mI`. Therefore

```
P_m = F_m^T F_m / m
P_m^2 = P_m = P_m^T
||F_m e||^2 / m = ||P_m e||^2 <= ||e||^2.
```

Use correlation units of 1/16 throughout:

```
H = sum_(u=1..L/2) e(u)^2 / 16
O = e(0)^2 / 16
G_m = ||F_m e||^2 / 16
Q = sum_(four parity groups) (sum x_group - required_sum)^2 / 4
B = sum_(all 2L signs) (1-x_i^2)^2

legacy   = H + O + 2 G_2 + 4 G_4 + 2 Q + B
balanced = H + O + G_2/2 + G_4/4 + 2 Q + B
```

Omit factor 4 when it does not divide L. Disabling compression removes both
folded terms. `binary_penalty` and `content_penalty` retain their existing
defaults 1 and 2; no per-length tuning is introduced.

Old folded weighting is `4||P_2 e||^2 + 16||P_4 e||^2` in these units.
Normalized weighting is `||P_2 e||^2 + ||P_4 e||^2`. The subspaces are nested
when 4 divides L, so the new terms still overlap; this is not whitening or
independent information. Normalization removes the factor-size amplification,
not all deliberate weighting. Whether that improves optimization is empirical.

There is also a conditioning bound **in symmetric residual space**, not in
the nonlinear sequence parameters. Since e(u)=e(L-u),

```
||e||^2/2 <= 16(H+O) <= ||e||^2.
```

For 4|L, the eigenvalues of the old correlation-only quadratic form lie in
`[1/2, 21]`; the normalized form lies in `[1/2, 3]`. Thus the respective
condition-number upper bounds are 42 and 6. With factor 2 alone the bounds
are 10 and 4. These are upper bounds, not measured condition numbers, and
ignore content/binary terms. Composing with quadratic PACF and tanh remains
nonconvex; no Hessian bound or convergence theorem for Adam follows from this.

## Distance to an exactly content-constrained binary pair

Let V be the finite set of binary sign pairs obeying the lane's four counts.
Define `D(x)=min_(z in V) ||x-z||^2`. Changing one assigned sign from +1 to -1
changes squared distance by `(x+1)^2-(x-1)^2=4x`. Thus in each group, assign
-1 to exactly the w smallest coordinates. This is the existing rank projection.

Away from ties, the selected vertex is locally constant and
`gradient D(x)=2(x-project(x))`. At a tie the loss is nondifferentiable; the
implementation consistently selects the lower-index branch. Detaching the
selected vertex is correct for this piecewise derivative; it is not a
straight-through derivative of quantized correlation.

```
projected = H + O + G_2/2 + G_4/4 + 2 Q + D
```

Parallel pairwise ranks give the same stable projection for short groups,
avoiding a stable sort in every MPS step. This uses O(batch*L^2) temporary
storage; it is an experimental option, not automatically faster than balanced.

For relaxed x in [-1,1], q=project(x), a conservative quantization-gap bound is

```
|C_q(u)-C_x(u)| <= 2 ||q-x||_1 <= 2 sqrt(2L D(x)).
```

This follows by expanding each changed product and using |x_i|,|q_i|<=1.
It explains what D controls. It is not a prediction of Hamming radius, a
safe-pruning rule or a guarantee that Adam will find a solution.

All losses are nonnegative. With positive binary weight, zero loss implies
binary signs (and correct content for D), while H and O enforce the full
target. Conversely an exact target pair with its matching content has zero
loss. Floating-point loss near zero is never accepted as proof: hard direct
PACF, original integer objective and the independent verifier still decide.

## Measurement protocol

`experiments/benchmark_torch_loss.py` uses fresh FKM seeds, identical initial
parameter fingerprints, identical Adam/cooling/restart settings, rotated
variant order, synchronized MPS wall budgets, and the same optional 250-step
GPU polish. All results stay in a new experiment directory. It never reads
existing PQCP files for initialization. C annealing and Z3 are not included
in this loss-only comparison, so it cannot establish end-to-end solution rate.

Reports compare the unchanged **discrete score** and original **fixed-target
integer energy**, not incomparable numeric values of different losses.
Initialization, formation, final observation and polish runtimes are separate.
Pytest uses only lightweight CPU cases; full MPS checks are explicitly enabled
when the device is available.

Old JSON checkpoints must retain `legacy` when `loss_mode` is absent. The new
mode is serialized, so resuming does not silently switch objective landscapes.

## MPS measurements (2026-09-10)

Initial exploration is in `results/torch_loss_pilot_20260910/`: one seed,
6 seconds per formation run, batch 2048, three lengths and three modes.
That pilot used a stable sort for D in every step. Replacing it with exact
parallel ranks improved projected-mode throughput, but did not make it the
best wall-clock choice. Do not combine its timings with the later implementation.

The main comparison is `results/torch_loss_comparison_20260910/runs.jsonl`:
real MPS, seeds 123/456/789, batch 2048 for ALL lengths, 12 seconds per
formation run, observation every 25 epochs, 2000-step restart horizon, common
250-step/8-candidate GPU-polish settings. Each pair's initial matrix fingerprint
matched. No known solutions supplied, no C or Z3. Formation totaled 325.41 s;
polish 22.09 s; initialization 9.26 s. Final observation is separately recorded.

| L | loss | median projected score | median score including polish | median exact target energy | median epochs |
|---|---|---:|---:|---:|---:|
| 44 | legacy | 12 | 12 | 26 | 1475 |
| 44 | balanced | 8 | 8 | 28 | 1464 |
| 44 | projected | 19 | 12 | 52 | 294 |
| 46 | legacy | 12 | 12 | 22 | 1550 |
| 46 | balanced | 12 | 12 | 16 | 1575 |
| 46 | projected | 16 | 12 | 26 | 311 |
| 68 | legacy | 53 | 46 | 133 | 883 |
| 68 | balanced | 32 | 32 | 93 | 875 |
| 68 | projected | 59 | 51 | 178 | 191 |

Balanced vs legacy: 6 lower best scores, 3 ties, 0 higher scores in nine
paired cases, both before and including polish. L44 target-energy median
actually increased slightly, despite better unrestricted score: the metrics
are not interchangeable. No verified PQCP was found by any mode in these
short tests. This is evidence of better short-budget candidate quality, NOT
evidence of a measured improvement in first-solution time or long-run yield.

`main.py` and `TorchHybridConfig` select **balanced** for fresh runs. The
lower-level `TorchSearchConfig` default stays legacy for API compatibility.
Use `--torch-loss legacy` to roll back a new main.py run; `projected` remains
an explicit experimental alternative because its per-step cost outweighed
its mathematical quantization-distance advantage on this Mac.

Follow-up at the **production L68 batch size 1024** is in
`results/torch_loss_L68_default_20260910/`, with the same three seeds and
15-second formation budgets (90.41 s total across six runs):

| seed | legacy projected / including polish | balanced projected / including polish |
|---|---:|---:|
| 123 | 38 / 38 | 32 / 32 |
| 456 | 45 / 45 | 20 / 20 |
| 789 | 48 / 32 | 37 / 37 |

Projected medians: 45 -> 32; including-polish medians: 38 -> 32. Balanced wins
all three projected comparisons, but only two of three after polish. This
counterexample matters: lower pre-polish score does not guarantee better
completion. Median epochs were 1325/1327, so the improvement is not from
giving the new loss more gradient steps. No verified solutions here either.

Polish uses the same settings, not a guaranteed equal number of proposals:
the unchanged adaptive archive filter selected 32--39 lanes in this follow-up.
`polish_input_lanes`, settings and duration are now explicitly recorded. The
primary fair comparison is equal-budget continuous formation; interpreting
post-polish scores as equal-swap-evaluation measurements would be incorrect.

Final validation: `.venv/bin/python -m pytest -q` on the actual MPS-capable
Mac: **699 passed**, 62.93 s. The only warning was the existing absence of
NumPy; no dependency was installed. Checks include real MPS value/gradient
agreement, finite-difference derivatives, exhaustive nearest-vertex and small
solution checks, residual conditioning bounds, legacy/new checkpoint replay,
default configuration wiring and existing solver regression tests.
All 39 retained best-candidate snapshots from the two main measurement sets
were additionally recomputed with the unchanged Python correlation, objective,
structured-energy and verifier modules; every stored value matched exactly.
