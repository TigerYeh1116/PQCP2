"""IDE Run: lift actual compressed witnesses and their full radius-two L1 balls.

C_A is fixed per anchor; G_A and all binary B bits are free. All strict
targets are searched. Compressed witnesses are not full PQCP solutions.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.complete_l86_full_rows import main


if __name__=='__main__':
    options=sys.argv[1:]
    if not options:
        for name in ('L86_actual_coarse_witnesses_radius2_complete_audit_20261008',
                     'L86_actual_coarse_witnesses_radius1_closed_20261008'):
            closed=ROOT/'results'/name
            if (closed/'certificate.json').exists():
                sys.argv+=['--resume-closed',str(closed)];break
    if not any(x.split('=')[0] in ('--input','--log','--coarse-input') for x in options):
        sys.argv+=['--coarse-input',str(ROOT/'results/L86_independent_norm_long_20261007/coarse_witnesses.json')]
    for positive,negative,value in (('--two-adic','--no-two-adic',None),
                                    ('--dihedral-support','--no-dihedral-support',None),
                                    ('--norm-unsat-cache',None,'4096'),('--unary-batch-free',None,'12'),('--top',None,'0')):
        if not any(x.split('=')[0] in (positive,negative) for x in options):
            sys.argv+=[positive]+([value] if value is not None else [])
    main(default_mode='coarse-row',default_radius=2)
