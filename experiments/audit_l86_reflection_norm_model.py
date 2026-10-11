"""IDE Run: full support/mod4 certificate for the symmetric norm-pair model."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solver.symmetric_norm_pair import symmetric_norm_support_pairs
from solver.checkpoint import atomic_write_json
from experiments.benchmark_l86_discovery import freeze_sources


def main():
    output=ROOT/'results'/('L86_reflection_norm_model_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False);freeze_sources(output/'source_snapshot')
    rows=[symmetric_norm_support_pairs(43,eta) for eta in (-1,1)]
    report=dict(complete=all(r['complete'] for r in rows),rows=rows,
        excluded_model='Both C rows individually reflection-symmetric, or both G rows individually reflection-symmetric. Consequently both binary rows individually reflection-symmetric.',
        universal_nonexistence_claim=False,new_discovery_speed_claim=False,
        claim='Restricted symmetric model only; no restriction added to arbitrary PQCP search.')
    atomic_write_json(output/'report.json',report)
    print(json.dumps(dict(output=str(output),complete=report['complete'],both_signs_excluded=all(r['excludes_entire_symmetric_model'] for r in rows))),flush=True)


if __name__=='__main__':main()
