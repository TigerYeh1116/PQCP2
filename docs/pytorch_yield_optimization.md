# PyTorch new-solution-rate optimization

## Success criterion and scope

The target is at least **2x new, independently verified, A/B-swap-deduplicated
PQCPs per wall-clock second**, not 2x Adam epochs, lower continuous loss or
lower best score. Equivalently, mean wall time per new pair must halve.
Time to first new pair is separately recorded; unsuccessful runs remain
censored and are never silently removed from a time average.

Project targets, `correlation.py`, integer objective, verifier, FKM, C moves,
temperature and Z3 algorithms are unchanged. MPS remains active concurrently
with compiled CPU completion. Existing verified pairs are not optimization
inputs; a frozen copy of the inventory is used only for fair deduplication.

## Low-risk paths, all independently switchable

1. **Parseval loss evaluation** (`loss_backend="spectral"`). With
   `R(f)=|X(f)|^2+|Y(f)|^2-[2L+8 eta cos(2pi fk/L)]`, periodic symmetry gives

   ```
   H+O = (sum_f multiplicity(f)*R(f)^2/L + e(0)^2 + e(L/2)^2)/32
   G_m = m/(16L) * sum_(m divides f) multiplicity(f)*R(f)^2
   ```

   The full/folded loss and penalties are mathematically identical to the
   previous implementation; only reconstruction and operation order differ.
   This alone gave little speedup: 300 L44/batch2048 Adam steps took 0.540 s
   using profile reconstruction, 0.522 s using Parseval, on actual MPS.
   Retained as an option, not evidence of 2x solution speed.

2. **Exact observation cache** (`fast_observation=True`). The existing
   verifier already calculates integer PACF, so reuse that result instead of
   recomputing it a second time. Repeated nonzero candidates reuse a bounded
   proof cache keyed by all ordered A/B bits and complete target profile.
   GPU score/profile/target energy still must agree with cached exact values.
   Score zero always invokes the independent verifier again. Cache capacity
   is 4*archive_size; it is transient, not checkpoint state. Metadata is
   refreshed and bit-step trajectory/archiving decisions remain unchanged.

3. **Exact Metal observation** (`observation_backend="metal"`). Two small
   kernels on PyTorch's MPS stream perform stable content ranking and integer
   PACF. Ranking counts coordinates strictly smaller than the current value,
   plus equal values at smaller indices. Thus it equals stable argsort,
   including ties, while retaining all four parity weights. PACF accumulates
   +/-1 products as int32 and writes exact float32 integers for L<=94.
   No loss gradients or annealing decisions are implemented in this module.
   Python recomputation remains the acceptance gate.

   Uses the installed [PyTorch 2.8 `torch.mps.compile_shader`
   API](https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html).
   No external extension, package installation or CPU fallback is needed.
   Requests outside even 4<=L<=94, float32 logits/int64 weights or actual MPS
   fail explicitly. Standard Torch observation remains available.

4. **Earlier first handoff** (`initial_formation_seconds=8`). Deliver the
   first candidate bank earlier while preserving persistent Adam and normal
   45-second later formation intervals. The CPU keeps running throughout.
   Its 15-second completion windows and SA decisions are not changed.
   This is a measured scheduling heuristic, not a mathematical pruning rule.

The runner also now stops/joins the C worker if MPS initialization fails,
instead of leaving a worker running outside the cleanup block.

## Measurement protocol

`experiments/benchmark_torch_yield.py` runs the real hybrid, not a mocked
solver. Wrappers around the real result writer record successful appends
after independently checking the candidate again. Every arm uses the same
frozen inventory, L, seed, wall budget, batch and SA settings. Variant order
alternates across seeds. Inventory/source fingerprints, full candidate A/B,
PACF, discovery time, source stage and counters are retained in a new isolated
directory. Python import time is excluded; algorithm initialization, FKM bank
construction, verification and shutdown are included. No Z3 is run in these
comparisons. No experimental discoveries are automatically fed back as seeds.

Variants:

- `current`: prior profile loss, normal observation, first handoff after 45 s.
- `spectral`: only Parseval evaluation.
- `cached`: only exact proof caching.
- `early`: only earlier first handoff.
- `optimized`: Parseval + caching + earlier handoff (first pilot).
- `metal`: original profile loss + caching + Metal observation + early handoff.
- `mixed`: `metal`, with 50% MPS and 50% existing FKM initialization for C.
- `feasibility`: `mixed`, changing only the continuous iteration to the
  experimental Douglas--Rachford method derived below.
- `feasibility_long`: the same projections, with 20,000 rather than 2,000
  steps per GPU restart (the existing stagnation limit scales accordingly).
- `wide`: `mixed`, with four times as many continuous Adam lanes. This does
  not change its loss, learning schedule, or C neighborhood.
- `frequency`: `mixed`, additionally removing profiles proved impossible by
  the quarter-frequency integer test below. Batch size remains 2048 at L44.

## Results so far (2026-09-10)

First pilot (`results/torch_yield_20260910_pilot`): L44, seeds 123/456/789,
90 seconds per arm, MPS batch 2048, same 4 C workers. Prior/current: 2 new
pairs in 277.167 s. Optimized: 2 in 276.038 s. Each arm succeeded for one of
three seeds. This is essentially a tie, **not a 2x improvement**. A favorable
individual seed was not treated as success, and the approach was not abandoned.

Then the actual observation bottleneck was addressed. Equal-work L44 tests
(1000 Adam steps, batch 2048, observation every 25 steps, original balanced
profile loss) gave:

| implementation | seconds | final best score |
|---|---:|---:|
| current | 6.158 | 8 |
| proof cache | 4.659 | 8 |
| proof cache + Metal | 2.090 | 8 |

This is **2.95x equal-work throughput**, not yet 2x new-solution rate.
Metal value equivalence, all small/large target lengths, stable ties and
Parseval gradients passed 51 actual-MPS focused tests before yield evaluation.

Follow-up protocol: fresh predetermined seeds 20260911/20260912/20260913,
L44, 120 seconds per arm, compare `current` and `metal`, with the original
profile loss held fixed. Artifacts: `results/torch_yield_20260910_holdout`.

Follow-up result: current found 3 new pairs in 364.832 s; Metal/early found
2 in 365.669 s (0.665x the new-pair rate, NOT an improvement). Both found a
first new pair in two of three runs. The median over successful runs alone
was 38.88 s versus 18.12 s; claiming 2x from those medians would improperly
discard censored failures and confuse first-hit time with new-pair rate.
All hits were produced by C completion. Some arrived before the first MPS
seed bank was consumed, so they cannot be attributed to MPS seed quality.

Faster formation raised epochs substantially but did not raise new-pair
yield. One plausible bottleneck is switching C to 100% of a bounded MPS
elite bank, losing FKM exploration; this is a hypothesis, not a proven cause.
The next `mixed` arm uses the same Metal/caching/early formation but feeds
50% MPS elites and 50% existing FKM initialization to C, using its existing
pair-seed selection control. MPS still runs continuously. No C neighborhood
or acceptance formula changes. The equal split is an exploration/exploitation
ablation, not a claimed optimal parameter.

The 180-second-per-arm mixed test is now complete, using fresh fixed seeds
20260914/20260915/20260916 (`results/torch_yield_20260910_mixed`). Current:
4 new pairs in 544.234 s (136.058 s/new). Mixed: 6 in 544.249 s
(90.708 s/new), approximately **1.50x**, below the requested 2x.
Paired counts were 1:1, 2:3, 1:2 (current:mixed). All six runs found a new
pair; mean first-hit time was 128.961 s versus 56.475 s. First-hit improvement
does not establish doubled sustained yield. Counts remain too small to
claim a reliable general speedup, and the mixed policy remains opt-in.

New options are wired to `main.py` but unproven policies are not silently
promoted to the one-click default:

```
--torch-fast-observation --torch-observation-backend metal
--mps-initial-seconds 8 --mps-pair-seed-percent 50
```

The first line enables the equivalent observation optimizations; the second
opts into the experimental early/mixed handoff. `--torch-loss-backend spectral`
separately selects equivalent Parseval evaluation. Default paths remain
available for rollback, and old checkpoints retain their original settings.

## Independent convergence experiment: feasibility projections

Reducing observation cost did not establish new-solution speed. An additional
**explicitly experimental algorithm**, not an equivalent loss rewrite, is
implemented in `torch_feasibility.py`. It adapts the feasibility approach in
[Aragon Artacho et al., *A feasibility approach for constructing combinatorial
designs of circulant type*](https://arxiv.org/abs/1711.02502). That paper applies
Douglas--Rachford to autocorrelation-defined combinatorial designs; it does
not establish convergence or speed for this PQCP implementation.

For our exact T, Fourier transformation gives
`T_hat(f)=2L+8 eta cos(2pi fk/L)>=0`. Let B be the binary pairs with the lane's
four parity counts. Let S be real pairs having that full PACF, additionally
with the lane's individual DC and Nyquist signed sums. Interior Fourier
coordinates `(Re A, Im A, Re B, Im B)` are projected radially onto a sphere
of radius `sqrt(T_hat(f))`. At DC/Nyquist, their individual real coordinates
are fixed from the four counts; imaginary coordinates vanish. Those endpoints
obey the necessary squared-sum identities already enforced by target profiles.
Parseval makes the separate frequency-block projections a nearest projection
onto S. Positive-sine analysis coordinates require positive-sine synthesis.
The zero-vector case has many nearest points; select positive real A and
zero B deterministically. No supplied or discovered PQCP supplies a phase.

The two-set iteration is

```
y = P_B(x)
z = P_S(2*y-x)
x_next = x + z - y
```

`P_B` is the exact content rank projection, optionally using the same Metal
kernel. The state x is a real matrix, not a binary answer. Only y is evaluated
with the original exact PACF and verifier. A fixed point has a binary shadow
in both sets, but the sets are nonconvex: cycling/stagnation remain possible.
The initial FKM batch, archive, restart controls, CPU completion and writer
are reused. No Adam moments are used in this mode; checkpoint validation
requires them to be empty and preserves the exact real iteration state.
The default remains the previous relaxed Adam method.

The first real-MPS/C feasibility comparison used L44, fresh fixed seeds
20260917/20260918/20260919, and 120 seconds per arm
(`results/torch_yield_20260910_feasibility`). Current found 5 new pairs in
364.276 s; feasibility found 1 in 363.622 s. That is **0.20x**, a negative
result, not evidence that projection methods cannot work. The single hit
also came from C completion, not a continuous projection being accepted as
a binary solution. In particular the inherited 2,000-step restart limit is
not validated for Douglas--Rachford. The paper's other design problems
sometimes require much longer trajectories; this motivates a separate
20,000-step ablation, not a claim that its results transfer to PQCP.

The extended comparison completed with new predetermined seeds
20260920/20260921/20260922, L44, 120 seconds per arm
(`results/torch_yield_20260910_extended`). Current: 6 new pairs/364.825 s.
Wide Adam: 1/366.791 s (0.166x). Longer feasibility: 4/364.198 s (0.668x).
Neither replacement is promoted. The first four completed studies' 25 new
pairs all came from C completion; their logs do not establish direct MPS
convergence or that every C hit came from an MPS-provided seed.

## Proven quarter-frequency initialization filter

`solver/frequency_constraints.py` is independent, integer-only mathematics.
The same mathematical condition already exists as the C kernel's opt-in
`PQCP_QUARTER_PRUNING` experiment; it was not wired into MPS initialization.
This change fills that gap rather than introducing a different condition.
The new Python implementation is cross-checked against the existing C probe;
no C source or environment switch is changed by this option.
For 4|L, put n=L/4. In a parity group with w ones, let c be the ones in its
first residue class modulo four. Its other residue class contains w-c ones,
so `max(0,w-n)<=c<=min(n,w)`. At frequency L/4, the half real/imaginary
Fourier coordinate equals `w-2c`. For all four A/B parity groups the target
therefore requires

```
sum_j (w_j - 2*c_j)^2 = L/2 + 2*eta*cos(pi*k/2).
```

Cosine here is the exact integer lookup 1,0,-1,0. Enumerating four finite
sumsets gives every attainable value, with no floating point. The four
groups are disjoint; discarding partial sums above the nonnegative target is
safe because remaining terms are nonnegative squares. Failure rules out
this **assigned target/content profile**, not different targets having the
same content. Success is only a necessary condition; verification is unchanged.
Lengths not divisible by four pass through without a conclusion.

At L44, 32 of the 128 existing decimation-reduced profiles are impossible.
Those 32 use k=2, eta=-1, with all four parity weights odd. The four squares
must sum to 4 mod 8, but the target is 24 = 0 mod 8: a contradiction. The
filter retains 96 profiles. At L68 it retains 104/128. At L46/86/94 it does
nothing. All retained profiles still initialize using the original FKM
procedure and random B; no known solution supplies seed bits.

Tests exhaustively check the parity-coordinate sumsets at L4/8/12/16 and
all binary pairs at L4/8, ensuring no actual solution target is rejected.
Filtered initialization and deterministic checkpoint continuation are also
tested. The opt-in CLI is `--torch-frequency-pruning`; its setting is saved
in checkpoints, and older checkpoints default to the prior unfiltered path.
The filter currently applies only to MPS profile initialization, not to the
C search or a hard threshold on moving candidates.

The paired end-to-end frequency test used fresh seeds
20260923/20260924/20260925, L44, 120 seconds per arm
(`results/torch_yield_20260910_frequency`). Current: 6 new pairs/364.069 s.
Frequency/mixed: 1/364.121 s (0.167x). Per-seed counts (current:frequency)
were 2:0, 0:1, 4:0. Safe elimination of impossible target families does not
by itself establish a stochastic search speedup. This composite arm also
includes the mixed/early/observation settings; it is not an isolated causal
measurement of the quarter condition. Defaults remain unchanged.

## Final verification of this measurement round

- Actual MPS full suite: **808 passed**, one existing missing-NumPy warning;
  no package installation was performed. Python/C quarter-condition equality
  is checked over every target profile at L4/8/12/44/46/68/86/94.
- 39 real hybrid trials, **4,924.092 s / 82.068 min** of measured execution.
  No trials with zero hits were dropped. No tests run a long benchmark.
- All **43 discovery events were distinct verified pairs**, all recorded by
  C completion. Each was rechecked against its frozen initial inventory and
  its run's final L.txt. Full exact profiles and score-zero checks passed.
- Two frozen inventory versions occurred across studies: 353 pairs in the
  pilot, 364 thereafter. Within each paired study the inventory was identical
  across methods/seeds. Do not pool different studies to estimate speedup.
- After all comparisons and verification finished, the 43 pairs were checked
  against the latest official inventory and appended with the existing
  atomic/verifying/A-B-swap-deduplicating writer. Official `44.txt` now has
  407 pairs. No supplied or discovered solution was used as an initializer.
  `results/torch_yield_verified_import.json` contains the full before-write
  text, before/after hashes and each appended pair for recoverability. Do not
  blindly restore that snapshot over any later discoveries.
- `results/torch_yield_optimization_audit.json` holds the full cross-study
  audit, candidate provenance and summaries. **The requested 2x new-solution
  speed is NOT demonstrated.** The best observed arm was the earlier mixed
  run at 1.50x, based on only four versus six discoveries; that is neither
  a doubled rate nor a reliable general improvement.

The remaining evidence gap concerns conversion into useful binary candidates,
not merely accelerator throughput. Larger batches and longer projections
were actually tested after initial weak results, and their negative results
remain available. Further work should separately examine quantization under
the stronger frequency/content conditions, then validate on new paired seeds;
it must not treat lower continuous loss or fewer impossible lanes as success.
