# Search recovery: L44 long-run regression (2026-09-09)

## Diagnosis from the actual run

`checkpoints/L44_torch.json`, seed 576, records 32,989.129 seconds,
7,496,000 Adam epochs, 157,356,032 projected candidate evaluations, and
**zero verified hits / zero new solutions**. The final best first appeared at
22,924.646 seconds and did not improve during the following 10,064.483 seconds.
Independent recomputation agrees with its stored PACF. Its objective is 4:
the value-distance penalty is zero but there are six nonzero sidelobes, not two:

```
u:       9  20  21  23  24  35
S(u):   -4  -4  +4  +4  -4  -4
```

This is not a verifier-passing pair lost during writing. The production entry
had changed algorithms: compressed FKM -> batch Adam -> rank quantization ->
verification, with no discrete SA/kick phase. The earlier C route instead runs
fixed-content same-parity swap SA, multiple proposals, a temperature schedule,
stagnation kicks and independent restarts. Epochs/projected evaluations are
not interchangeable with SA moves or unique binary candidates. MPS continuous
loss and the displayed discrete objective are also different quantities.

Small MPS optimizations/experiments are retained, not declared impossible:
DFT sped up the continuous kernel ~3.08x, but a 12-run L44/L46 pilot found zero
new pairs. Exact MPS swap polish, four legal proposals, kicks and compression
ablations also failed to find a solution in those bounded probes. More steps
or a lower score alone were therefore insufficient recovery evidence.

## Minimal production fix

`main.py` now defaults to the already-existing **C compiled swap-SA backend**,
with its existing compressed FKM bank, two sampled legal moves, eight workers,
mathematical constraints and result handling unchanged. Clicking Run still
asks for L, seed and optional Z3, and searches until Ctrl+C. Verification and
exact/A-B-swap deduplication still precede appending to `L.txt`.

This is an explicit restoration of the proven discrete search route, **not a
claim that pure PyTorch/Adam has been made equally effective**. C uses CPU;
`--backend mps` still uses actual MPS and never silently falls back. MPS code,
experiments and old checkpoints remain available. Resuming a Torch checkpoint
still resumes Adam and prints that fact; it does not convert optimizer/RNG
state into a C run. To use restored production search, start `main.py` fresh.
No mathematical definition, verifier, C search algorithm or `.gitignore` was
changed for this recovery. No environment/package installation was needed.

## Actual new-solution measurements

All rows use L44, eight C workers, two sampled proposals, existing compressed
FKM initialization, and Z3 off. Every recorded hit passed the independent
verifier and was absent from the prior project inventory including A/B swaps.

First, the retained production C kernel was tested with three seeds that had
also been used in unsuccessful MPS runs, with a 300-second cap and stopping
at the first genuine new pair:

| Seed | First new PQCP (seconds) |
|---|---:|
| 576 | 16.326 |
| 8543 | 6.172 |
| 856 | 7.469 |

Mean: 9.989 seconds; 3/3 hit, 3 distinct new verified pairs. This timing includes
C process startup and verification but excludes bank preparation. Full source,
seeds, candidate bits and PACFs: `results/search_recovery_L44_first_new.json`.

Then the **real no-argument main.py** entry was tested on three additional seeds.
The harness supplies exactly the interactive inputs `L`, `seed`, `no Z3`.
Timing starts before Python process creation and includes initialization,
seed-bank preparation, the search and independently checked file writing.
It then sends Ctrl+C and checks normal exit. Each run starts with an isolated
copy of the current official inventory; verified new discoveries are published
back to the project afterwards.

| Seed | First new, including startup (seconds) | New verified pairs | Exit |
|---|---:|---:|---:|
| 90210 | 13.494 | 1 | 0 |
| 20260909 | 36.815 | 1 | 0 |
| 2026 | 20.248 | 1 | 0 |

Mean: 23.519 seconds; all under the user's five-minute reference. Six distinct
new L44 pairs across the two measurements were appended to `44.txt`.
Artifacts: `results/search_recovery_actual_entry/` (source snapshots, per-seed
terminal logs, complete candidate bits/PACF, timings and result inventories).
Reusable harness: `experiments/benchmark_search_recovery.py`. It is opt-in;
pytest never runs this 300-second search benchmark.

Final regression: `.venv/bin/python -m pytest -q` on the actual MPS-capable
Mac environment: **611 passed**, no skips, 57.03 seconds. The one warning is
the already-existing Torch warning about missing NumPy, not a test failure.
The restored interactive default, explicit MPS selection, original MPS
checkpoint routing, integer math, verifier, deduplication and C counters are
covered. A separate final audit recomputed every one of the six new pairs,
confirmed objective zero and membership in the now-287-pair official `44.txt`,
and confirmed the measured C source hash was unchanged. The original MPS
checkpoint still contains elapsed 32,989.129, epoch 7,496,000 and seed 576.

## Scope and remaining risks

The reported failure was the actual L44 run above; the recovery is measured
on L44, not extrapolated to L46/68/86/94 or to every random seed. Stochastic
search cannot promise a solution within five minutes on every run. This is
not evidence of any length's impossibility. The pure MPS path remains an
experimental alternative; future MPS/SA integration must compare complete
time-to-new-verified-solution before becoming the default again.
