"""Nearest fixed-count four-color point, certified using exact dyadic integers.

Native Hungarian assignment proposes primal/dual variables. Its floating point
dual is accepted only if ALL dual inequalities and chosen equalities hold in
exact integer arithmetic. Otherwise a Python integer Hungarian solve repairs it.
"""
import ctypes
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


@lru_cache(maxsize=1)
def _library():
    root = Path(__file__).resolve().parents[1]
    source = root/'csrc/colour_projection.c'
    build = root/'build'; build.mkdir(exist_ok=True)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = build/('colour_projection_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+suffix)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=build, suffix=suffix); os.close(fd)
        try:
            subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', temporary],
                check=True, capture_output=True, timeout=30)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    lib = ctypes.CDLL(str(path))
    dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.pqcp_colour_assignment.argtypes = [dp, dp, ctypes.c_int, ip, ip, dp, dp]
    lib.pqcp_colour_assignment.restype = ctypes.c_int
    return lib


def _scaled(values):
    ratios = [float(v).as_integer_ratio() for v in values]
    scale = max(d for _, d in ratios)
    return [n*(scale//d) for n, d in ratios]


def certify_assignment(c, g, slots, assignment, u, v):
    n = len(c)
    if (len(assignment) != n or sorted(assignment) != list(range(n)) or len(u) != n or len(v) != n
            or any(not math.isfinite(x) for x in tuple(u)+tuple(v))):
        return False
    exact = _scaled(tuple(c)+tuple(g)+tuple(u)+tuple(v))
    c, g, u, v = (exact[i*n:(i+1)*n] for i in range(4))
    maxima = {color: max(v[j] for j in range(n) if slots[j] == color) for color in set(slots)}
    for i, j in enumerate(assignment):
        costs = (-c[i], c[i], -g[i], g[i])
        if u[i]+v[j] != costs[slots[j]] or any(u[i]+dual > costs[color] for color, dual in maxima.items()):
            return False
    return True


def _integer_assignment(c, g, slots):
    n = len(c)
    exact = _scaled(tuple(c)+tuple(g)); c, g = exact[:n], exact[n:]
    costs = [tuple((-c[i], c[i], -g[i], g[i])[color] for color in slots) for i in range(n)]
    u, v, p, way = [0]*(n+1), [0]*(n+1), [0]*(n+1), [0]*(n+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minv, used = [None]*(n+1), [False]*(n+1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = None; j1 = 0
            for j in range(1, n+1):
                if used[j]: continue
                current = costs[i0-1][j-1]-u[i0]-v[j]
                if minv[j] is None or current < minv[j]: minv[j], way[j] = current, j0
                if delta is None or minv[j] < delta: delta, j1 = minv[j], j
            for j in range(n+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                elif minv[j] is not None: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    assignment = [0]*n
    for j in range(1, n+1): assignment[p[j]-1] = j-1
    return assignment


def project_colour_row(c, g, counts):
    c, g, counts = tuple(c), tuple(g), tuple(counts)
    n = len(c)
    if (not 1 <= n <= 47 or len(g) != n or len(counts) != 4 or sum(counts) != n
            or any(type(k) is not int or k < 0 for k in counts)
            or any(not math.isfinite(float(v)) for v in c+g)):
        raise ValueError('finite coordinates and four integral counts of total length required')
    slots = tuple(color for color, count in enumerate(counts) for _ in range(count))
    raw_c, raw_g = (ctypes.c_double*n)(*c), (ctypes.c_double*n)(*g)
    assignment, u, v = (ctypes.c_int*n)(), (ctypes.c_double*n)(), (ctypes.c_double*n)()
    status = _library().pqcp_colour_assignment(raw_c, raw_g, n, (ctypes.c_int*4)(*counts), assignment, u, v)
    certified = status == 0 and certify_assignment(c, g, slots, assignment, u, v)
    selected = tuple(assignment) if certified else tuple(_integer_assignment(c, g, slots))
    labels = tuple(slots[j] for j in selected)
    if tuple(labels.count(color) for color in range(4)) != counts:
        raise RuntimeError('nearest-color assignment changed contents')
    return labels, certified
