"""Paired fresh GPU batches with/without integer navigation; isolated result root.

Additional native work is measured separately. This is not an equal-wall-time
discovery-rate experiment and never reads existing PQCPs as search seeds.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solver.torch_hybrid_runner import TorchHybridConfig,_direct_compression_round
from solver.torch_device import resolve_torch_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='mps',choices=('cpu','mps','cuda'))
    parser.add_argument('--seed',type=int,default=1919810)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--top',type=int,default=4)
    parser.add_argument('--moves',type=int,default=131072)
    args=parser.parse_args()
    if min(args.rounds,args.steps,args.top,args.moves)<1:parser.error('work limits must be positive')
    device=resolve_torch_device(args.device)
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    configs={top:TorchHybridConfig(86,seed=args.seed,root=args.output,direct_compression=True,
        direct_compression_steps=args.steps,direct_compression_native_top=top,
        direct_compression_native_moves=args.moves, direct_compression_support_top=0) for top in (0,args.top)}
    for cycle in range(args.rounds):
        # Alternate host order to reduce systematic warmup/order effects.
        for top in ((0,args.top) if cycle%2==0 else (args.top,0)):
            cfg=configs[top];_direct_compression_round(cfg,device,cycle)
            path=args.output/'results/direct_compression'/cfg.direct_compression_session/f'L86_seed{args.seed}_cycle{cycle:06d}.json'
            row=json.loads(path.read_text());row['native_top']=top;rows.append(row)
            print(json.dumps(dict(cycle=cycle,top=top,gpu_energy=row['minimum_gpu_compressed_energy'],
                energy=row['minimum_compressed_energy'],seconds=row['elapsed'],
                native_seconds=row['native_seconds'],native_improved=row['native_improved'],exact_jobs=row['exact_jobs'])),flush=True)
            (args.output/'report.json').write_text(json.dumps(dict(config={k:str(v) if isinstance(v,Path) else v
                for k,v in vars(args).items()},rows=rows),indent=2))


if __name__=='__main__':main()
