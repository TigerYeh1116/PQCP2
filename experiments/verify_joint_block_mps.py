"""Audit every retained row option and EVERY Gram score on a real L86 block."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from solver.checkpoint import atomic_write_json
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf
from solver.target_profiles import TargetContentProfile,pair_content
from solver.torch_joint_blocks import exact_joint_block
from solver.torch_joint_search import binary_joint_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('Real MPS required')
    args.output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((ROOT/'results/L86_GS_interleave_20261007/report.json').read_text())['seeds'][0]
    words=tuple(map(tuple,saved['words']))
    target=TargetContentProfile(86,saved['k'],1,*pair_content(*words))
    c,g=binary_joint_rows(words,target)
    positions=((0,4,8,12,16,20,24,28),(1,5,9,13,17,21,25,29))
    rows,scores,groups=0,0,0
    def audit(residual,da,db,matrix,oa,ob):
        nonlocal rows,scores,groups
        exact=[]
        for side,options in enumerate((oa,ob)):
            old=tuple(v for row in (c[side],g[side]) for v in pacf(row)[1:22])
            values=[]
            for option in options:
                labels=dict(zip(positions[side],option))
                cs=tuple((1 if labels[i]==0 else -1 if labels[i]==1 else 0) if i in labels else v for i,v in enumerate(c[side]))
                gs=tuple((1 if labels[i]==2 else -1 if labels[i]==3 else 0) if i in labels else v for i,v in enumerate(g[side]))
                profile=tuple(v for row in (cs,gs) for v in pacf(row)[1:22])
                values.append(tuple(x-y for x,y in zip(profile,old)))
            exact.append(torch.tensor(values,dtype=torch.int64))
            rows+=len(options)
        if da.dtype!=torch.int32 or db.dtype!=torch.int32 or matrix.dtype!=torch.int32:
            raise RuntimeError('precision-critical block evaluation must use integer tensors')
        if not torch.equal(da.cpu(),exact[0].to(torch.int32)) or not torch.equal(db.cpu(),exact[1].to(torch.int32)):
            raise RuntimeError('MPS deltas differ from independent literal row PACF')
        left=exact[0]+torch.tensor(residual,dtype=torch.int64)
        expected=left.square().sum(-1)[:,None]+exact[1].square().sum(-1)[None,:]+2*(left@exact[1].T)
        if not torch.equal(matrix.cpu(),expected.to(torch.int32)):
            observed=matrix.cpu()
            difference=observed-expected.float()
            bad=(difference!=0).nonzero()[:8].tolist()
            atomic_write_json(args.output/'precision_rejection.json',dict(
                content_group=groups,rows=(len(oa),len(ob)),mismatches=int((difference!=0).sum()),
                maximum_absolute_error=float(difference.abs().max()),
                first_bad=[dict(index=index,observed=float(observed[tuple(index)]),
                                expected=int(expected[tuple(index)])) for index in bad],
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                max_input=(int(left.abs().max()),int(exact[1].abs().max())),
                reason='Rejected before any discovery/pruning performance claim.'))
            raise RuntimeError('MPS Gram matrix differs from independent integer row-profile energy')
        scores+=matrix.numel();groups+=1
    started=perf_counter()
    result=exact_joint_block(words,target,positions,device=torch.device('mps'),matrix_audit=audit,temperature=1.)
    report=dict(target=asdict(target),positions=positions,row_options_audited=rows,
        pair_energies_audited=scores,content_groups=groups,result=asdict(result),elapsed=perf_counter()-started,
        complete=True,scope='ALL retained row options and ALL pair scores of this fixed L86 block; not discovery-speed evidence.')
    atomic_write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='result'}),flush=True)
    print('minimum full energy',result.minimum_joint_energy//2,'zero candidates',len(result.zero_words))


if __name__=='__main__':main()
