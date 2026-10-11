"""An exact unsigned-pair counting bound for odd factor-two PQCP targets.

For supports U,V of total weight d, let W=1-V, so |W|=|U|=w.
P_U+P_W must be odd at d//2-1 independent shifts (all except k).
The sum of those unsigned counts is 2*binomial(w,2)=w*(w-1).
Hence each original support must satisfy w*(w-1)>=d//2-1.
"""


def minimum_support_size(d):
    if type(d) is not int or d < 3 or not d % 2:
        raise ValueError('odd half-length >=3 required')
    required = d//2-1
    weight = 0
    while weight*(weight-1) < required:
        weight += 1
    return weight


def support_sizes_meet_bound(weights, d):
    if len(weights) != 2 or any(type(w) is not int or not 0 <= w <= d for w in weights) or sum(weights) != d:
        raise ValueError('two support sizes of total weight d required')
    lower = minimum_support_size(d)
    return all(w >= lower for w in weights)
