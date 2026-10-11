"""IDE Run: radius-two signed C_A anchors, all G_A signs and all B bits free.

Uses the latest root L86 long-test log, eight audited non-solution pairs.
All strict target positions/signs and complete necessary support families
are retained. Every acceptance passes the original full binary verifier.
Default norm lifts, bounded SINGLE-norm UNSAT cache and support reflection
are enabled; exact graph-cut projection remains optional.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_l86_full_rows import main


if __name__=='__main__':main(default_mode='coarse-row',default_radius=2)
