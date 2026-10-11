"""Exact paired-block C/G enumeration with necessary count pruning.

Keep every coordinate outside two selected antipodal blocks fixed. Enumerate
all local four-color words preserving row even/odd counts, and pair only
contents satisfying total C weight d plus existing proven content bounds.
Integer PACF deltas reduce every pair energy to an exact int32 Gram matrix.
Greedy block minima are exact. Positive-temperature navigation is heuristic;
all zero candidates remain independently audited regardless of navigation.
"""
from dataclasses import asdict,dataclass
from functools import lru_cache
from itertools import combinations
import math
import random

import torch

from .correlation import full_correlation_profile
from .mps_memory import autorelease_pool
from .structured_energy import integer_periodic_autocorrelation_profile as pacf, target_profile
from .target_profiles import target_content_profiles
from .torch_compressed_search import search_paired_contents
from .torch_joint_search import binary_joint_rows,joint_binary_words
from .verifier import verify_pqcp

C_VALUES=(1,-1,0,0)
G_VALUES=(0,0,1,-1)


@lru_cache(maxsize=256)
def color_permutations(counts):
    if (len(counts)!=4 or any(type(v) is not int or v<0 for v in counts)
            or not 1<=sum(counts)<=8):
        raise ValueError('four nonnegative color counts of total 1..8 required')
    remaining=list(counts)
    prefix,words=[],[]
    def visit():
        if len(prefix)==sum(counts):
            words.append(tuple(prefix));return
        for color in range(4):
            if not remaining[color]:continue
            remaining[color]-=1;prefix.append(color)
            visit()
            prefix.pop();remaining[color]+=1
    visit()
    return tuple(words)


def block_options(c,g,positions):
    """ALL local words preserving the two row parity counts, by new global Nc."""
    labels=tuple(0 if c[p]==1 else 1 if c[p]==-1 else 2 if g[p]==1 else 3 for p in positions)
    m=len(labels)
    local_nc=labels.count(1)
    even=local_nc+labels.count(3)
    odd=local_nc+labels.count(2)
    outside_nc=c.count(-1)-local_nc
    groups={}
    for nc in range(max(0,even+odd-m),min(even,odd)+1):
        counts=(m-even-odd+nc,nc,odd-nc,even-nc)
        groups[outside_nc+nc]=color_permutations(counts)
    return groups


def block_row_deltas(c,g,positions,options,*,device):
    """Full independent C/G PACF deltas, without constructing full candidates."""
    d,m=len(c),len(positions)
    if torch.device(device).type=='mps':
        from .mps_joint_block import row_deltas
        return row_deltas(c,g,positions,options)
    pairs=tuple(combinations(range(m),2))
    incidence=[[int((positions[j]-positions[i])%d==u or (positions[i]-positions[j])%d==u)
                for u in range(1,d//2+1)] for i,j in pairs]
    kernel=torch.tensor(incidence,dtype=torch.int32,device=device).reshape(len(pairs),d//2)
    result=[]
    for row,values in ((c,C_VALUES),(g,G_VALUES)):
        changes=torch.tensor([[values[color]-row[p] for p,color in zip(positions,option)] for option in options],
                             dtype=torch.int32,device=device)
        linear=torch.tensor([[row[(p+u)%d]+row[(p-u)%d] for u in range(1,d//2+1)] for p in positions],
                            dtype=torch.int32,device=device)
        delta=changes@linear
        if pairs:
            products=torch.stack([changes[:,i]*changes[:,j] for i,j in pairs],-1)
            delta=delta+products@kernel
        result.append(delta)
    return torch.cat(result,-1)


def _replace_block(c,g,positions,labels):
    c,g=list(c),list(g)
    for p,color in zip(positions,labels):c[p],g[p]=C_VALUES[color],G_VALUES[color]
    return tuple(c),tuple(g)


def joint_integer_energy(words,target):
    profile=tuple(full_correlation_profile(*words))
    wanted=target_profile(target.L,target.k,target.eta)
    value=sum((x-y)**2 for x,y in zip(profile[1:target.L//2+1],wanted[1:target.L//2+1]))
    if value%16:raise RuntimeError('binary target energy lattice failed')
    return value//8  # twice the original full energy


@dataclass
class JointBlockResult:
    words: tuple
    selected_joint_energy: int
    minimum_words: tuple
    minimum_joint_energy: int
    zero_words: list
    configurations: int
    content_groups: int
    complete: bool


class JointBlockAuditError(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message)
        self.details=details


@torch.no_grad()
def exact_joint_block(words,target,positions,*,device,temperature=0.,seed=123,
                      matrix_audit=None,zero_callback=None):
    if (target not in target_content_profiles(target.L,decimation_reduced=False)
            or not 6<=target.L<=94 or target.L%4!=2
            or len(positions)!=2 or any(not 1<=len(p)<=min(8,target.L//2) or len(set(p))!=len(p)
                or any(type(i) is not int or not 0<=i<target.L//2 for i in p) for p in positions)
            or type(seed) is not int or not isinstance(temperature,(int,float))
            or isinstance(temperature,bool) or not math.isfinite(temperature) or temperature<0):
        raise ValueError('valid Project target, two bounded antipodal blocks and temperature required')
    c,g=binary_joint_rows(words,target)
    positions=tuple(map(tuple,positions))
    allowed=search_paired_contents(target)
    allowed_nc={(a[0],b[0]) for a,b in allowed}
    if (c[0].count(-1),c[1].count(-1)) not in allowed_nc:
        raise ValueError('center violates proven content bounds')
    d=target.L//2
    full=target_profile(target.L,target.k,target.eta)
    tc=tuple((full[u]+full[u+d])//4 for u in range(d))
    tg=tuple((-1 if u%2 else 1)*(full[u]-full[u+d])//4 for u in range(d))
    pc,pg=(tuple(x+y for x,y in zip(pacf(pair[0]),pacf(pair[1]))) for pair in (c,g))
    residual=tuple(x-y for x,y in zip(pc[1:d//2+1],tc[1:d//2+1]))+tuple(x-y for x,y in zip(pg[1:d//2+1],tg[1:d//2+1]))
    r=torch.tensor(residual,dtype=torch.int32,device=device)
    options=[block_options(c[side],g[side],positions[side]) for side in (0,1)]
    rng=random.Random(seed)
    minimum=chosen=None
    min_value=None
    min_ties=0
    reference=partition=None
    zeros=[]
    configurations=groups=0

    def reconstruct(labels_a,labels_b):
        changed=[_replace_block(c[side],g[side],positions[side],labels) for side,labels in enumerate((labels_a,labels_b))]
        return joint_binary_words(tuple(v[0] for v in changed),tuple(v[1] for v in changed),target)

    for nc_a,nc_b in sorted(allowed_nc):
        if nc_a not in options[0] or nc_b not in options[1]:continue
        oa,ob=options[0][nc_a],options[1][nc_b]
        with autorelease_pool(device):
            da=block_row_deltas(c[0],g[0],positions[0],oa,device=device)
            db=block_row_deltas(c[1],g[1],positions[1],ob,device=device)
            left=da+r
            if torch.device(device).type=='mps':
                from .mps_joint_block import pair_scores
                scores=pair_scores(da,db,r)
            else:
                scores=left.square().sum(-1,dtype=torch.int32)[:,None]+db.square().sum(-1,dtype=torch.int32)[None,:]+2*(left@db.T)
            if matrix_audit is not None:matrix_audit(residual,da,db,scores,oa,ob)
            configurations+=scores.numel();groups+=1
            value=float(scores.min().item())
            if value<0 or not value.is_integer():raise RuntimeError('Gram matrix failed integer energy bound')
            value=int(value)
            locations=(scores==value).nonzero().cpu().tolist()
            count=len(locations)
            pick=locations[rng.randrange(count)]
            local_min=(oa[pick[0]],ob[pick[1]])
            if min_value is None or value<min_value:
                min_value,minimum,min_ties=value,local_min,count
            elif value==min_value:
                if rng.randrange(min_ties+count)<count:minimum=local_min
                min_ties+=count
            if value==0:
                for ia,ib in locations:
                    pair=reconstruct(oa[ia],ob[ib])
                    check=verify_pqcp(*pair)
                    if not check.is_valid or tuple(check.profile)!=full:
                        raise RuntimeError('block zero failed independent original-length verifier')
                    zeros.append(pair)
                    if zero_callback is not None:zero_callback(pair)
            if temperature:
                weights=torch.exp((value-scores)/(2*temperature))
                mass=float(weights.sum().item())
                new_reference=value if reference is None else min(reference,value)
                if partition is not None:partition*=math.exp((new_reference-reference)/(2*temperature))
                group_mass=mass*math.exp((new_reference-value)/(2*temperature))
                if partition is None or rng.random()<group_mass/(partition+group_mass):
                    cumulative=weights.flatten().cumsum(0)
                    cutoff=rng.random()*float(cumulative[-1].item())
                    slot=int((cumulative>cutoff).to(torch.int32).argmax().item())
                    chosen=(oa[slot//len(ob)],ob[slot%len(ob)])
                reference,partition=new_reference,(0. if partition is None else partition)+group_mass
    if minimum is None:raise RuntimeError('block enumeration omitted its legal original center')
    best=reconstruct(*minimum)
    actual=joint_integer_energy(best,target)
    if actual!=min_value:
        raise JointBlockAuditError('block minimum failed independent original binary energy audit',dict(
            reference=words,target=asdict(target),positions=positions,minimum_labels=minimum,
            minimum_words=best,reported_joint_energy=min_value,actual_joint_energy=actual,
            reason='Rejected; no UNSAT or discovery claim from this block.'))
    selected=best if not temperature or min_value==0 else reconstruct(*chosen)
    selected_energy=joint_integer_energy(selected,target)
    if min_value>joint_integer_energy(words,target):
        raise RuntimeError('block minimum excluded the original center')
    return JointBlockResult(selected,selected_energy,best,min_value,zeros,configurations,groups,True)
