# PyTorch MPS PQCP search

**Current default:** `main.py` uses unseeded, persistent MPS population
refinement for every length, followed by compiled same-parity completion.
Existing `L.txt` pairs are used only by the output writer for deduplication;
they are not input seeds or training examples. See
[unseeded refinement](pytorch_unseeded.md) for the current design and tests.
The material below is historical experimental documentation, including older
default choices. `--backend adam` and explicit Adam checkpoints retain that
older workflow; `--backend c` remains a rollback for isolated comparisons.

## Historical experiments

Production recovery (2026-09-09): clicking Run on `main.py` now uses the retained
compiled swap-SA pipeline again. The pure MPS/Adam default had produced no
verified hit during a recorded 32,989-second L44 run. See
[search recovery](search_recovery.md) for the measured diagnosis and recovery.
This page describes the **explicit MPS experiment**, not the one-click default.
All PyTorch search operations still require actual MPS; no CPU fallback was added.

```sh
python main.py --L 44 --seed 123                  # production C swap SA
python main.py --backend mps --L 44 --seconds 20 --torch-batch-size 512
python main.py --resume checkpoints/L44_torch.json
python main.py --L 44 --backend c                 # previous compiled SA
python main.py --backend mps --L 44 --torch-kernel dft  # experimental continuous PACF
```

For the MPS experiment, MPS must actually be available. No CPU fallback is allowed.
PyTorch is already installed in this project's `.venv`; no NumPy is required
by this implementation. `torch.backends.mps.is_available()` must return True.
Use float32 on MPS. A Colab Linux GPU is not an Apple MPS device; this entry
is specifically for the Mac. CPU is an explicit library-only testing mode.
See [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.8/notes/mps.html).
The PACF backward uses the analytic derivative
`d rho(x,u)/d x[p] = x[p+u] + x[p-u]`, evaluated by reductions on MPS.
This avoids the generic gather backward's repeated-index atomic accumulation,
whose order otherwise caused tiny numerical differences on checkpoint replay.

## Reference adaptation

Reviewed the supplied `奇數長度PACP搜尋.pptx` and `偶數長度PACP搜尋.pptx`, including
their flow diagrams, and Project 2 question 3 on page 29 of `文獻/06_PACP.pdf`.
The following ideas are reused: a batch of continuous sequence parameters,
Adam with cyclic cosine learning rates, binary quantization, top-candidate
archive, and restarting stagnant trajectories with fresh Adam moments.

The odd PACP loss `(|S(u)|-2)^2` and even PACP half-shift-only target are NOT
used. Neither is the slides' larger equivalence relation: formal output uses
only exact pair equality and A/B exchange, as required by this project.

For signs x=(-1)^A and y=(-1)^B, the PQCP target is T(0)=2L,
T(k)=T(L-k)=4 eta for 0<k<L/2, and zero elsewhere. Two *actual shifts* count
as two sums, but represent one independent symmetric degree of freedom.

Existing `target_profiles.py` provides necessary DC/Nyquist constraints:

```
sum(x)^2 + sum(y)^2 = 2L + 8 eta
Alt(x)^2 + Alt(y)^2 = 2L + 8 eta (-1)^k
```

Each batch lane selects one feasible target/parity-content profile. A starts
from the existing genuine factor-2 compressed FKM lifts (i and i+L/2); B is
randomized with exact even/odd weights. The batch visits profile families
using a seeded shuffled cycle. Initial noise uses a private generator.

Continuous signs are tanh(theta/temperature). Write e=S-T. The following is
the retained **legacy** per-lane loss. Fresh `main.py` runs now default to
`balanced`, dividing folded terms by 2 and 4 instead of multiplying by 2 and
4; see [the mathematical derivation and measurements](torch_loss_math.md).
Exact objective, archive ranking and verifier are unchanged.

```
E_full = sum_{u=1}^{L/2} e(u)^2 / 16
E_m = sum_{r=0}^{L/m-1} (sum_q e(r+q L/m))^2 / 16
loss = E_full + 2 E_2 + 4 E_4 + e(0)^2/16
       + 2 * sum_{A/B,parity}(relaxed sign sum - required sign sum)^2 / 4
       + sum_{A/B,i}(1-sign_i^2)^2
```

E_4 is omitted unless 4 divides L. `--torch-full-only` disables E_2/E_4 for
ablation. These weights retain the old multiscale navigation convention;
they are heuristics, not a theorem about speed. Loss is not the displayed
integer `pqcp_objective` score and does not certify a solution.

Every 25 epochs by default, all 512 lanes are projected to the exact four
even/odd one counts using device-side rank sorting. All discrete scores are
evaluated on MPS. Every zero-score lane and the top 8 are fully recomputed on
CPU with existing integer correlation and the independent verifier. A valid
pair at a different k is still saved. An archive retains up to 64 candidates.
Optional completion reuses the existing asynchronous guidance/beam/Z3 bridge
and its existing guards; selecting Z3 does not mean every elite calls Z3.

The batch restarts after 1000 updates, or if every lane has stalled for at
least 300 updates at the discrete observation points. Each rebirth resets
parameters, Adam moments, temperature and learning-rate cycle. Batch size is
bounded and configurable, rather than automatically growing to 120,000 on
shared Mac memory. A restart counts one lane; an epoch updates the whole
batch. `projected evaluations` counts quantized lanes scored, not SA swaps.

## Persistence and verification

- New verified pairs append to `L.txt` in the existing format, with A/B-swap
  deduplication and the existing writer lock.
- Best candidate: `results/best/L44_torch_seed123.json`.
- Improvement history: `logs/torch_L44_seed123.jsonl`, including A/B and S(u).
- Safe JSON checkpoint: `checkpoints/L44_torch.json`, containing float32
  parameters, Adam first/second moments, optimizer step, target profiles,
  iteration/restart state, best/elite candidates and algorithm parameters.
- Resume restores the optimizer without new FKM initialization. Same-device
  continuation is tested; bitwise agreement across Torch versions or between
CPU and GPU is not promised. Random initialization is derived solely from
seed and batch generation, so no hidden global Torch RNG is consumed.

## Optional real-DFT kernel and discovery measurement

The default `direct` kernel remains a rollback/baseline. `--torch-kernel dft`
uses real matrix multiplication only for continuous PACF/loss/gradients.
For real signs (or relaxed real values), write `X[f]=sum_i x[i] exp(-2πifi/L)`.
Then `rho(u)=sum_f |X[f]|² cos(2πfu/L)/L`. Negative frequencies duplicate the
positive frequencies, except 0 and L/2. Thus a small real cosine/sine matrix
replaces the large batch-by-shift-by-position gather intermediate. It does
not zero-pad or change cyclic length. The DFT constants are constructed once
on CPU in double precision and moved to MPS float32; optimization remains on
MPS, with no complex FFT fallback. The mathematical loss is unchanged, but
float32 summation order can change the trajectory. The exact direct kernel
still handles **all** discrete scores and solution detection, and full Python
correlation/verifier remain the persistence authority. Checkpoints save the
kernel choice; older checkpoints without it restore `direct`.

`experiments/benchmark_mps_acceleration.py` separates two questions:

```sh
python experiments/benchmark_mps_acceleration.py --mode throughput \
  --output results/mps_kernel_trial
python experiments/benchmark_mps_acceleration.py --mode discovery \
  --lengths 44 46 --seconds 60 --seeds 123 456 789 \
  --output results/mps_discovery_trial
```

Output must be a new directory to avoid overwriting evidence. Throughput
timing alternates methods and synchronizes MPS before/after each sample.
Discovery comparisons use equal wall-clock budgets including initialization,
freeze the same pre-experiment L.txt inventory for both methods, and dedup
across all seeds within each method. All newly discovered pairs are independently
verified and also published through the existing project's writer. Each method
keeps its own inventory, so earlier publication cannot penalize the other arm.
Actual elapsed time (including final persistence), source hashes, configurations,
per-run pairs and summaries are recorded. A zero baseline gives an undefined
rate ratio, **not** infinite improvement. Three times optimizer throughput
does not demonstrate three times new-PQCP discovery rate.

### Measured acceleration pilot (2026-09-09, project Mac)

Batch 512, real MPS, alternating method order. Pure optimizer timing uses
three repetitions of 100 updates per L after warmup:

| L | Direct updates/s | DFT updates/s | Core throughput ratio |
|---|---:|---:|---:|
| 44 | 194.5 | 598.8 | 3.079 |
| 46 | 201.3 | 619.6 | 3.079 |

Artifacts: `results/mps_acceleration_kernel_512/`.

The discovery pilot uses seeds 123, 456, 789, each method/L/seed with a
60-second budget including initialization. Actual total wall time including
final persistence was 722.413 seconds. No concurrent GPU search/test was run;
brief CPU-only correctness tests ran during part of the pilot, so this is
not a fully isolated performance certification.

| L | Direct best scores | DFT best scores | End-to-end update-rate ratio | New pairs, direct / DFT |
|---|---|---|---:|---:|
| 44 | 8, 12, 15 | 8, 12, 8 | 2.111 | 0 / 0 |
| 46 | 8, 12, 8 | 8, 11, 8 | 2.215 | 0 / 0 |

Artifacts (full candidates, seed logs, frozen inventories, summaries):
`results/mps_acceleration_discovery_60s/`. These observations prove neither a
threefold discovery rate nor failure of the DFT approach. Core acceleration
survives in the real loop, but initialization, observation and persistence
reduce it. No successful time-to-solution observation was obtained.

### Experimental discrete polish (not enabled in main.py)

`solver/torch_polish.py` implements the slides' batched discrete finishing idea,
using PQCP targets and **same-parity, opposite-sign swaps**. Every two-flip
delta includes the shared-edge correction; both shared edges count at L/2.
All integer profiles remain exact on MPS and returned candidates are fully
recomputed before they can be published. Fixed-target energy navigates, but
the saved best uses the unrestricted PQCP integer objective, retaining solutions
at a different k. Lanes freeze on the first zero within this bounded block.
Random uniforms are explicit inputs. Main search and its checkpoints are not
changed by this standalone, opt-in probe.

```sh
python experiments/benchmark_mps_acceleration.py --mode polish \
  --checkpoints results/mps_acceleration_discovery_60s/L44/direct/checkpoints/L44_torch.json \
                results/mps_acceleration_discovery_60s/L44/dft/checkpoints/L44_torch.json \
  --steps 8000 --elites 64 --clones 8 --output results/mps_polish_trial
```

An initial exploratory 2,000-step geometric-temperature 4-to-0.1 probe found
no solutions and accepted under 1% of legal swaps. Rather than discard the
approach, the default was aligned with the existing C temperature schedule:
`64*(6*(1-step/steps)+0.15)` in raw squared-error units is
`24.6 - 24*step/steps` in this module's energy/16 units. This changes a heuristic
schedule, not PQCP conditions. Original probe source snapshots are retained.

With 64 recorded elites and eight independent random draws per elite
(512 lanes), 8,000-step probes gave:

| L | Device polish time | Legal swaps | Accepted swaps | Best before / after | New verified pairs |
|---|---:|---:|---:|---|---:|
| 44 | 11.131 s | 1,961,995 | 83,450 | 8 / 4 | 0 |
| 46 | 9.834 s | 1,951,258 | 184,658 | 8 / 8 | 0 |

Results and all returned candidates are in
`results/mps_polish_probe_L44_c_temperature/` and the corresponding L46
directory. This excludes the cost of generating those elites and is **not**
an end-to-end speedup comparison. The earlier 200%-faster MPS discovery goal
was not established by these probes. Further checks include multiple-proposal selection (as in existing C),
elite/restart diversity and equal-total-budget integration, without changing
the independent verifier or substituting scores for real discoveries.

Full regression after these changes, on actual MPS with the project `.venv`:
`python -m pytest -q` → **575 passed**, no skips, 55.82 seconds. One existing
Torch warning reports absent NumPy; these implementations do not require it.
Tests include direct/DFT value and gradient agreement, identical FKM starts,
same-device DFT checkpoint replay, old-checkpoint compatibility, exact swap
deltas including half-shift overlap, per-step parity/PACF checks on CPU/MPS,
other-target-shift solution retention and benchmark zero-baseline handling.

```sh
python -m pytest -q tests/test_torch_search.py
python experiments/benchmark_mps_search.py --L 44 --seconds 20
```

MPS-specific tests skip when the execution environment cannot access Metal.
Do not interpret those skips as GPU validation. Continuous convergence or
more candidates per second alone does not prove faster discovery of new PQCPs.

## Initial validation on the project Mac

PyTorch 2.8.0, float32, batch 512, seed 123, 20-second runs (completion off):

| L | Initial discrete score | Best discrete score | Epochs | New verified PQCP |
|---|---:|---:|---:|---:|
| 44 | 105 | 16 | 4282 | 0 |
| 46 | 99 | 8 | 4075 | 0 |

Parameter and gradient devices were both `mps:0`. Reports retain full A/B,
integer correlation, configuration, wall time and progression in
`results/torch_mps_L44_smoke.json` and `results/torch_mps_L46_smoke.json`.
These are short integration measurements, not evidence of a speedup over C.

Validation also included the full regression suite (509 passed, 11 MPS tests
skipped inside the restricted environment), followed by 82 targeted tests
passing with actual Metal access after the final persistence changes. A real
CLI SIGINT saved epoch 70 and a subsequent CLI resume continued to epoch 86.
The MPS entry with optional completion enabled also exited successfully in a
temporary workspace. One existing Torch warning concerns absent NumPy; this
search does not depend on NumPy.
