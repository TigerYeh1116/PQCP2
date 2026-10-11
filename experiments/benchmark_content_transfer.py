"""Replay saved references for diagnosis, never seed production or write L.txt."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.target_profiles import TargetContentProfile
from solver.ternary_content_transfer import repair_content_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/direct_compression'))
    parser.add_argument('--pattern', default='L86_seed123_repair*.json')
    parser.add_argument('--max-energy', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = []
    started = perf_counter()
    for path in sorted(args.input.glob(args.pattern)):
        for record in json.loads(path.read_text())['attempts']:
            if record['original_energy'] > args.max_energy:
                continue
            target = TargetContentProfile(**record['target'])
            result, work = repair_content_transfer(*record['reference'], target)
            attempts.append(dict(source=str(path), lane=record['lane'], reference=record['reference'],
                                 target=record['target'], energy=record['original_energy'],
                                 result=asdict(result), work=work))
    report = dict(config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  elapsed=perf_counter()-started, attempts=attempts)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(attempts=len(attempts), elapsed=report['elapsed'],
                         sat=sum(r['result']['status']=='SAT_COMPRESSED' for r in attempts))))


if __name__ == '__main__':
    main()
