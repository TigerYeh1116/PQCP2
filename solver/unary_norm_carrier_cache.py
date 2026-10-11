"""Bounded in-process reuse of audited C_B norm support domains.

Only proven C_B-UNSAT supports disappear. Every known SAT carrier and every
unresolved support remains available; G results never exclude a support
at a different target. No checkpoint/JSON cache import is supported.
"""
from collections import OrderedDict

from .structured_energy import integer_periodic_autocorrelation_profile as pacf


class UnaryNormCarrierCache:
    def __init__(self,slot_budget=4096):
        if type(slot_budget) is not int or not 1<=slot_budget<=65536:
            raise ValueError('bounded support slot budget 1..65536 required')
        self.slot_budget=slot_budget;self._items=OrderedDict();self.slots=0;self.hits=0

    def _get(self,key):
        if key not in self._items:return None
        value=self._items.pop(key);self._items[key]=value;self.hits+=1
        return value

    def _remember(self,key,candidates,proof):
        if not candidates or len(candidates)>self.slot_budget:return False
        profile,negative=key;d=len(profile);seen=set();checked=[]
        for mask,word in candidates:
            if type(mask) is not int or not 0<=mask<1<<d or mask in seen:
                raise RuntimeError('carrier cache requires unique literal support candidates')
            if bin(mask).count('1')!=profile[0]:raise RuntimeError('carrier/deferred support weight differs from the exact single norm')
            seen.add(mask)
            if word is not None:
                word=tuple(word)
                if (len(word)!=d or any(type(v) is not int or v not in (-1,0,1) for v in word)
                        or tuple(pacf(word))!=profile or word.count(-1)!=negative
                        or sum((v!=0)<<j for j,v in enumerate(word))!=mask):
                    raise RuntimeError('cached C_B carrier failed independent literal norm/count/support audit')
            checked.append((mask,word))
        if not proof['all_c_b_supports_accounted'] or proof['omitted_unknown_supports']!=0:
            raise RuntimeError('support-domain cache may omit only completely proven C_B-UNSAT supports')
        if key in self._items:self.slots-=len(self._items.pop(key)['candidates'])
        while self._items and self.slots+len(checked)>self.slot_budget:
            _,old=self._items.popitem(last=False);self.slots-=len(old['candidates'])
        self._items[key]=dict(candidates=tuple(checked),proof=dict(proof));self.slots+=len(checked)
        return True
