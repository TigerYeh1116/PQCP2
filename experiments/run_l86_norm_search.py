"""Press IDE Run: one million fresh support groups, exact shared-C/G audit.

The proven phase-count method is enabled; --no-phase-count-pruning rolls back.
All trials, including UNKNOWN, remain in a compressed replayable trace.
Each run gets its own timestamped directory and never overwrites old evidence.
"""
from datetime import datetime
from pathlib import Path
import sys
import secrets

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.search_norm_supports import main

RUN_TRIALS=1000000
RUN_SEED=None  # Set an integer here to replay; None records a fresh seed per Run.


if __name__=='__main__':
    options=sys.argv[1:]
    defaults=(('--output',str(ROOT/'results'/('L86_independent_norm_phase_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))),
              ('--trials',str(RUN_TRIALS)),('--seed',str(secrets.randbelow(2**31) if RUN_SEED is None else RUN_SEED)))
    for option,value in defaults:
        if not any(arg.split('=')[0]==option for arg in options):sys.argv.extend([option,value])
    if not any(arg.split('=')[0] in ('--gzip-trace','--no-gzip-trace') for arg in options):sys.argv.append('--gzip-trace')
    if not any(arg.split('=')[0] in ('--projection-groups','--no-projection-groups') for arg in options):sys.argv.append('--projection-groups')
    if (not any(arg.split('=')[0] in ('--weight-prefix','--no-weight-prefix') for arg in options)
            and not any(arg.split('=')[0] in ('--no-projection-groups','--no-shared-c-completion') for arg in options)):
        sys.argv.append('--weight-prefix')
    main()
