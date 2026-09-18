# MPS-heavy PQCP search

The default `main.py` path now assigns most of each search cycle to PyTorch
MPS and keeps the compiled CPU annealer as a completion stage:

1. Build independent relaxed matrices for the admissible Project 2
   `(k, eta, ordinary sum, alternating sum)` target profiles. No known PQCP is
   used as training data or initialization.
2. Optimize 2048 matrices at once through L=58, or 1024 for wider lengths.
   The real DFT correlation kernel evaluates the same periodic correlation as
   the direct definition but uses a smaller MPS workspace.
   The default continuous loss now uses normalized folded energies
   `E_2/2 + E_4/4`, instead of `2 E_2 + 4 E_4`. This changes only continuous
   guidance, not exact archive ranking, CPU/GPU annealing, score or verifier.
   Derivation and controlled comparisons: [torch_loss_math.md](torch_loss_math.md).
3. Rank-project every lane to an exact binary pair with the required number of
   ones in each sequence and parity class.
4. Rank candidates by exact integer fixed-target energy, not by continuous
   loss. Keep at least the best 32 candidates, with additional candidates only
   inside an adaptive energy window.
5. Run the existing exact same-parity GPU swap polish for 250 steps. This
   preserves all four content counts.
6. Give the resulting close seed bank to the existing compiled C annealer for
   15 seconds. Every C restart uses a seed from this bank.
7. Recompute every possible solution with the independent Python verifier
   before the existing deduplicating `L.txt` writer is allowed to append it.

The default cycle is 45 seconds of continuous MPS formation plus GPU polish,
then 15 seconds of compiled completion. The search continues until Ctrl-C when
no time budget is supplied.

The continuous loss and compressed energies are navigation heuristics only.
They never replace the exact objective, Project 2 definition, or verifier.

Useful rollback/ablation controls:

- `--backend adam` retains the earlier standalone Adam path.
- `--backend c` retains the compiled-only path.
- `--torch-batch-size N` overrides the length-aware lane count.
- `--torch-loss legacy` restores the original continuous loss;
  `--torch-loss projected` selects the experimental fixed-content projection
  distance variant. Existing checkpoints retain their saved loss mode.
- `--mps-seconds-per-cycle S` and `--cpu-seconds-per-cycle S` alter only the
  wall-time allocation.
