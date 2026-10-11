"""IDE Run entry: two bounded changed halves, a third entirely free.

No command line is required. Uses the latest root L86 long-test log,
eight audited non-solution centers, and total flip radius three. Results
have frozen sources and exact declared-region completion evidence.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_two_l86_halves import main


if __name__=='__main__':main(default_varied_halves=2,default_radius=3)
