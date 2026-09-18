# Unseeded PyTorch MPS refinement

The default `main.py` flow now uses the same from-scratch process for all
lengths. It never loads an existing PQCP as a seed, teacher or parent. The
existing independent writer still reads `L.txt` for exact/A-B-swap deduplication.
Previously the default depended on whether that file contained a valid pair,
and the no-known-solution fallback discarded its population after one call to
`step()`. That call updated logits which were never used to generate offspring.

## What learns and what stays exact

`solver/torch_population.py` retains complete non-solution A/B pairs for each
target/content island, as well as learned per-position logits. Each generation
uses 75% offspring of those intact pairs and 25% freshly sampled candidates.
Half the fresh samples ignore logits, so learned concentration cannot eliminate
exploration. The remaining samples follow the learned fixed-content Gumbel
distribution. Targets and four parity weights come from `target_profiles.py`;
initial FKM biases use genuine compressed FKM lifts, not a result inventory.

Offspring ordinarily receive one legal opposite-bit, same-parity swap; every
eighth generation uses two and every 32nd uses four. Selection preserves
distinct low-energy parent pairs per island. These are seed-generation
heuristics, not new mathematical conditions or safe pruning. The constants
are explicit controls, not claimed optimal parameters. Actual solutions are
excluded from parent selection, probability learning and the completion bank.
Degenerate all-solved tiny islands may retain inert slots but cannot mutate
them into new seeds; independent random candidates continue to be sampled.

Offspring PACF updates reuse `torch_polish.swap_correlation_delta`: for each
nonzero shift, the two individual flip deltas are added and their shared edge
terms corrected. At `u=L/2` both directed shared edges count; `u=0` never
changes. Thus the incremental part costs O(L) per swap. No continuous
relaxation/rounding is used to decide a discrete candidate's score. Float32
additions of small integer correlations and energies are exact at the listed
Project lengths, with full Python integer recomputation at the output boundary.

Navigation retains the existing fixed-target energy
`E_full + 2 E_2 + 4 E_4` (in units of 1/16; `E_4` only when 4 divides L).
`pqcp_objective` and Project 2 page 29 remain unchanged: two actual nonzero
shift indices, a symmetry pair, each of magnitude 4. A low energy is not a
solution certificate. Independent verifier PASS is mandatory.

## Production integration

`solver/torch_hybrid_runner.py` keeps one population object alive across refresh
cycles. Its logits, candidate archive and private RNG advance rather than
resetting every cycle. Defaults: two islands per canonical profile, population
128, archive 16 per island, up to 128 generations or 8 seconds per MPS phase.
The time cap is checked between generations; initialization and atomic export
can extend wall time. All tensor generation, mutation, PACF evaluation and
selection execute on MPS in production. CPU is an explicit test setting.

Each MPS score-zero discovery is independently verified and appended through
the existing writer immediately. Nonzero archive elites are exported round-robin
over target islands to avoid starving a target family when a global top-K is
dominated by another family. The bank is atomically replaced and a JSON sidecar
contains exact A/B, S(u), score and target metadata. Best snapshots are saved to
`results/best/L{L}_mps_unseeded_seed{seed}.json` immediately on improvement,
with a complete A/B/profile JSONL history under `logs/mps_unseeded_L*_seed*.jsonl`.

C retains the existing mixed initialization policy: 25% of restarts use the
MPS bank and 75% use the existing FKM/spectral starts. Refined seeds enter C
unchanged (`pair_seed_swaps=0`); the old unconditional 1..3 preliminary swaps
could throw away the local structure MPS had just improved. The C SA kernel,
target, acceptance and verifier are retained. Optional Z3 still receives C
elites through the existing asynchronous bridge.

The output keeps cumulative C `restart / moves / swap evaluations / valid
results` across cycles, plus a separate MPS line with generation and discovery
counts. MPS generations are not mislabeled as C restarts. Selecting Run on
`main.py` and entering L starts this flow. Explicit raw-Adam checkpoints retain
their old trajectory; they are not converted to this population representation.

## Reproducible first-discovery measurement

`experiments/benchmark_mps_unseeded.py` compares:

- `bootstrap`: repeatedly discard the one-generation model, reproducing the
  previous no-learning behavior.
- `cem`: reuse the previous probability-only learning method.
- `refined`: keep intact pairs, incremental swaps and fresh exploration.

The benchmark uses the same population/profile family and wall budget, rotates
method order, synchronizes the device, and creates an empty directory per run.
It does not copy official results or checkpoints. No C/Z3 is used, so successful
discovery measures PyTorch's contribution directly. Every zero is verified and
stored in the isolated run's `L.txt`. Tests use small bounded generations;
pytest never executes this timed experiment. Failed runs are marked censored;
their time cap is not presented as an observed solution time or a speedup.

```sh
python experiments/benchmark_mps_unseeded.py --lengths 28 44 68 \
  --seeds 1024 2026 4096 --seconds 8 \
  --output results/my_unseeded_comparison
```

Artifacts contain configurations, source hashes, per-run JSON, JSONL history,
complete improvement candidates, verifier results and group summaries. A
separate optional C completion can be measured through the production runner;
it is not conflated with pure-PyTorch discovery times.

MPS device/timing APIs follow the installed PyTorch backend. See the official
[MPS notes](https://docs.pytorch.org/docs/2.8/notes/mps.html) and
[synchronization API](https://docs.pytorch.org/docs/2.8/generated/torch.mps.synchronize.html).

## Confirmation experiment on this Mac

`results/mps_unseeded_confirmation_20260909/` records 27 isolated real-MPS
trials, seeds 1024, 2026, 4096, up to eight seconds per trial including
initialization (generation-boundary stopping allows a small overrun). Total
actual time: 190.795 seconds. The model shape is identical for all three arms:
two islands per target family, population 128, 16 elites. No known solutions,
C, Z3 or concurrent search/test were used. Benchmarking stopped at the first
generation containing a verified solution, retaining all zeros in that batch.

| L | One-generation bootstrap best scores | Persistent CEM best scores | Refined best scores |
|---|---|---|---|
| 28 | 4, 4, 4 | 0, 4, 4 | 0, 0, 0 |
| 44 | 34, 30, 34 | 22, 26, 24 | 8, 8, 8 |
| 68 | 143, 142, 151 | 85, 75, 82 | 27, 40, 43 |

At L=28, refined first-verification times were 0.698, 0.309 and 0.350 seconds:
3/3 successes, four distinct pairs within the per-run inventories. Bootstrap
was 0/3; CEM was 1/3 at 4.343 seconds. These are experimental small lengths,
not a replacement for the Project's listed lengths. At L=44/68 all arms were
0/3: improved scores are evidence about candidate refinement, **not proof of
improved first-solution time at those lengths**. No finite speedup ratio is
claimed for a baseline with no observed solution.

The preceding pilot in `results/mps_unseeded_pilot_20260909/` used seeds
123, 456, 789 and five-second budgets. It already showed 3/3 versus 0/3
L=28 successes against CEM; the confirmation added uniformly sampled
immigrants and used different seeds. The `validation` directory contains an
aborted startup caused by a fixed parameter-name collision, not a completed
trial; it is excluded from every table. Source changes during confirmation
were limited to type annotations; sampled moves and selection code were not
changed. The manifest records the source hash loaded for the experiment.
