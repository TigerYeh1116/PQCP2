# MPS-heavy search path

`main.py` now uses the following production path by default:

1. Compressed-FKM supplies structurally valid, fixed-content initial points.
2. PyTorch keeps a persistent batch of continuous `(A, B)` matrices on MPS.
3. The continuous loss uses the Project 2 target assigned to each lane and
   the existing full/factor-2/factor-4 residual energies. The default
   `balanced` loss normalizes each folded squared norm by its factor, using
   `E_full + E_origin + E_2/2 + E_4/4` plus the unchanged content/binary
   penalties. Omit factor 4 if it does not divide L. Exact integer ranking
   and annealing energy are unchanged. See [the derivation](torch_loss_math.md);
   `--torch-loss legacy` restores the former continuous weights.
4. Every 25 Adam epochs, rank projection produces exact binary sequences with
   the required even/odd content. Accelerator values are not trusted as a
   solution verdict.
5. Candidates are ranked by their exact integer energy to their assigned
   `(k, eta, content)` target. The archive keeps both this value and the
   original target-independent `pqcp_objective` score.
6. The closest projected candidates receive 250 steps of the existing
   same-parity, opposite-sign swap annealer on MPS.
7. At least the closest 32 candidates, and at most 128 candidates, are passed
   to the compiled C annealer. C uses this bank for every restart in that
   completion window.
8. A score-zero candidate is always recomputed by the pure-Python correlation
   implementation and accepted only if the independent verifier passes.

MPS formation and compiled C completion run concurrently. C starts immediately
from compressed-FKM seeds; whenever MPS publishes a new immutable cycle bank,
the next C subprocess consumes that bank. C subprocesses use at most four
threads while MPS is active: on the target Mac this preserved more MPS epochs
and matched the aggregate throughput measured with eight competing C threads.
With no time option, search remains unbounded until Ctrl-C. The default
continuous batch is 2048 for lengths up to
58 and 1024 for longer lengths; measurements showed that wider matrices need a
smaller batch to retain enough gradient updates per second. Both values can be
overridden with `--torch-batch-size`.

This path does not read an existing `L.txt` to form or train seeds. `L.txt` is
consulted only by the existing result writer for exact/A-B-swap deduplication.
The old discrete-population former and raw Adam backend remain available as
rollback/ablation code, but they are not the one-click default.
