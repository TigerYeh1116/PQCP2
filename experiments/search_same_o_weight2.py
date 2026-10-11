"""IDE entry: ALL 21 impulse displacements, arbitrary O, target centered +/-1."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.search_same_o_weight4 import main

if __name__=='__main__':main(default_g_weight=2)
