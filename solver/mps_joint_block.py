"""Typed integer Metal kernels for exact paired-block energy evaluation."""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _library():
    source=Path(__file__).resolve().parents[1]/'csrc/joint_block.metal'
    return torch.mps.compile_shader(source.read_text())


def row_deltas(c,g,positions,options):
    d,m=len(c),len(positions)
    h=d//2
    device=torch.device('mps')
    labels=torch.tensor(options,dtype=torch.int32,device=device)
    old=torch.tensor([[row[p] for p in positions] for row in (c,g)],dtype=torch.int32,device=device)
    linear=torch.tensor([[[row[(p+u)%d]+row[(p-u)%d] for u in range(1,h+1)] for p in positions]
                         for row in (c,g)],dtype=torch.int32,device=device)
    indices=torch.tensor(positions,dtype=torch.int32,device=device)
    shape=torch.tensor((len(options),m,h,d),dtype=torch.int32,device=device)
    output=torch.empty((len(options),2*h),dtype=torch.int32,device=device)
    _library().pqcp_block_deltas(labels,old,linear,indices,shape,output,threads=output.numel())
    return output


def pair_scores(da,db,residual):
    if (da.dtype!=torch.int32 or db.dtype!=torch.int32 or residual.dtype!=torch.int32
            or da.ndim!=2 or db.ndim!=2 or residual.ndim!=1
            or da.shape[1]!=db.shape[1] or da.shape[1]!=len(residual)):
        raise ValueError('compatible int32 row-delta and residual vectors required')
    na,nb,width=len(da),len(db),len(residual)
    shape=torch.tensor((na,nb,width),dtype=torch.int32,device=da.device)
    output=torch.empty((na,nb),dtype=torch.int32,device=da.device)
    _library().pqcp_block_scores(da.contiguous(),db.contiguous(),residual.contiguous(),shape,output,
        threads=(((nb+15)//16)*16,((na+15)//16)*16),group_size=(16,16))
    return output
