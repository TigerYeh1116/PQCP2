# C search optimization: measurement record

Goal: reduce the time required to find a genuinely new, independently verified
Project 2 PQCP to less than one fifth of the pre-change production search.
**The goal is not yet demonstrated.** Moves/second alone is insufficient.

## Reversible implementations

`csrc/pqcp_search.c` keeps its original arithmetic under the compile flag
`-DPQCP_REFERENCE_KERNEL`. This selects the original swap evaluation and
per-move atomic counters. The default uses exact, optimized arithmetic:

- For signs x and flipped positions p,q, at u != 0:
  `delta = -2*x[p]*(x[p+u]+x[p-u]) - 2*x[q]*(x[q+u]+x[q-u])`
  `        + 4*x[p]*x[q]*([p+u=q] + [q+u=p])`.
  Indices are cyclic. Both directed-edge corrections apply at u=L/2.
- Store periodic copies during evaluation to remove inner-loop modulo.
- With h=L/2 and symmetric residual e, factor-two residuals are
  `v[r]=e[r]+e[h-r]`; factor four is another fold of v. The exact existing
  `E_full + 2*E_2 + 4*E_4` is retained, omitting factor four when unavailable.
- Reuse the selected swap's integer correlation deltas on acceptance.
- Batch observational counters per thread (1024 moves and at restart end).
  Live counts can lag; the joined final counts are exact.
- Replace remainder operations in the original random sampler with an exact
  reciprocal calculation. Every original draw/rejection remains unchanged.

Runtime variants (the C binary uses explicit environment switches):

- `PQCP_QUENCH=1`: up to eight strict downhill steps from a restart best whose
  energy is at most 256, exhaustively inspecting the existing legal swaps.
- `PQCP_MOVE_POOL=1`: index positions by parity and sign and sample directly
  from the concatenation of the two legal Cartesian products. Every unordered
  legal swap occurs exactly once. The lists update in O(1) after an accepted
  swap or kick. This consumes different RNG draws and needs statistical
  comparisons; it is not a seed-by-seed trajectory-preserving change.
  The Python C backend now selects this sampler by default through
  `CSearchConfig.direct_swap_sampling`. `main.py --legacy-sampler` restores
  the rejection sampler; direct callers can set `direct_swap_sampling=False`.
- `PQCP_QUARTER_PRUNING=1`: for L divisible by four, exclude parity contents
  whose four quarter-frequency squared components cannot sum to the target.
  If a parity class has w ones and length L/2, splitting its ones over the
  two mod-4 classes gives `d=w-2*c`, with `|d|<=min(w,L/2-w)` and parity w.
  The necessary identity is `sum(d_j^2)=L/2+2*eta*cos(pi*k/2)` for the four
  A/B parity classes. Each d varies independently within those exact bounds.
  No candidate in a rejected content family can meet the target; same-parity
  swaps keep the family fixed. For L=44,k=2,eta=-1, counts (9,11,7,11) force
  four odd squares, summing to 4 modulo 8, whereas the target is 24 (0 modulo
  8). Counts (8,12,8,10) pass. The rule is inactive when 4 does not divide L.
  Exhaustive actual-sequence spectral cross-checks are in test_c_kernel.py.

Quench and quarter-frequency pruning remain opt-in experiments.

Elites carry a full-recomputed C objective as metadata. The Python bridge can
skip its existing score rejections without recomputing every ineligible
profile. Candidates proceeding to completion still have their score checked
against independent Python full recomputation. The handoff thresholds remain
unchanged.

## Verification

`tests/test_c_kernel.py` compares C against full Python correlation/energy at
lengths 4 through 94, checks exact uint64 remainder boundaries, and compares
fixed-work reference/optimized trajectories. It exhaustively checks the move
pool's complete legal-neighborhood coverage for lengths 4,6,8, and replays
hundreds of pool swaps with invariant checks. The quarter-frequency condition
is compared with all actual sequence spectra at lengths 4,8,12. Latest full
suite: **482 passed** (one existing Torch warning). Default/legacy-sampler CLI
smokes passed, as did a Z3-enabled CLI smoke with one actual Z3 execution.

## Measurement protocol

`experiments/benchmark_c_kernel_search.py` builds reference and optimized C
with the same compiler flags. It runs methods sequentially in counterbalanced
order with identical length, master seed, threads, FKM seed bank, and budget.
The C sampler remains fixed at two candidate swaps per move. It does not
supply known solutions as seeds. It records source hash and configuration.

Every reported hit is independently checked using `verifier.py`, deduplicated
under exact equality and A/B exchange, and checked against the preexisting
`L.txt` inventory. Full A/B and correlation are retained in the experiment's
JSON file. Official L.txt files are not changed during paired measurements.
No-hit trials are censored; do not average only successful first-hit times and
call that an unconditional mean. These trials measure C search with Python
verification; end-to-end Z3-enabled throughput still needs a separate check.

## Findings so far

- `results/c_kernel_pool_throughput.json`: 3-second paired runs at L=44/46,
  seeds 123/456, eight threads. Direct-pool throughput is about 37.7--38.0M
  moves/s vs reference 5.76--6.31M. This is approximately 6--6.5x throughput,
  but none of these very short trials found a new solution.
- `results/c_kernel_discovery_holdout.json`: 60 seconds per trial, three
  seeds per method and length. Reference vs optimized-plus-quench yielded
  L=44: 0 vs 3 new hits; L=46: 1 vs 0. This does not establish a speedup and
  does not justify enabling quench by default. The route remains available
  for larger and combined experiments.
- `results/c_kernel_pool_discovery.json`: completed five-seed, 60-second
  reference/direct-pool measurement. L=44: reference 0 new pairs in 300s,
  pool 10 distinct new pairs in 300s (observed exposure/new-pair ratio 30.0s).
  L=46: both methods 0 new pairs in 300s each. Sustained throughput ratios
  were 5.70x (44) and 5.14x (46). With zero control discoveries, these data
  still do not establish the requested fivefold solution-time improvement.

Verified discoveries from these studies and the pilot were imported through
the production verifier/deduplicating writer: 14 new pairs in 44.txt and one
in 46.txt. `results/c_kernel_verified_import.json` retains the receipt and
complete before-write text for rollback. The paired timing studies themselves
ran against frozen initial inventories; imports took place after completion.

Next evidence required: enough verified discoveries across independent seeds
to assess actual time per new solution, then a full production pipeline check
with the chosen Z3 mode and persistence. Do not mark the goal complete based
on an arithmetic benchmark or an isolated successful seed.
