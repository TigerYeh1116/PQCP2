"""IDE Run entry: all four physical halves may change, with exact completion.

Three halves each have at least one flip within a total radius-four ball;
the fourth is entirely free. Uses eight audited centers from the newest
root L86 log. Each declared fixed-three fiber is fully decided and audited.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_two_l86_halves import main


if __name__=='__main__':main(default_varied_halves=3,default_radius=4)
