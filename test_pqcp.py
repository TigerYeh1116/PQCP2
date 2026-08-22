import time
import z3


# ============================================================
# Parameters
# ============================================================

L = 44


# ============================================================
# Periodic autocorrelation
# ============================================================

def periodic_autocorrelation(seq, shift):
    """
    rho(seq; shift)
    for a binary {0,1} sequence.

    rho(seq; u)
        = L - 2 * HammingDistance(seq, shifted_seq)
    """

    return L - 2 * z3.Sum([
        z3.If(
            z3.Xor(seq[i], seq[(i + shift) % L]),
            1,
            0
        )
        for i in range(L)
    ])


# ============================================================
# Create binary sequences
# ============================================================

a = [z3.Bool(f"a_{i}") for i in range(L)]
b = [z3.Bool(f"b_{i}") for i in range(L)]


# ============================================================
# Create solver
# ============================================================

solver = z3.Solver()

# Give Z3 a reasonable time limit.
solver.set(timeout=120_000)


# ============================================================
# Calculate S(u) = rho(a;u) + rho(b;u)
# ============================================================

S = [
    periodic_autocorrelation(a, u)
    + periodic_autocorrelation(b, u)
    for u in range(L)
]


# ============================================================
# Basic condition
#
# At u = 0:
#
# rho(a;0) = L
# rho(b;0) = L
#
# therefore:
#
# S(0) = 2L
# ============================================================

solver.add(S[0] == 2 * L)


# ============================================================
# Exactly TWO nonzero autocorrelation sums
#
# Because periodic autocorrelation satisfies
#
# S(u) = S(L-u),
#
# two nonzero shifts must form one symmetric pair:
#
#       u  <---->  L-u
#
# Therefore we select exactly ONE pair.
# ============================================================

pair_selected = [
    z3.Bool(f"pair_{u}")
    for u in range(1, L // 2)
]


# Exactly one pair is selected.
solver.add(
    z3.Sum([
        z3.If(p, 1, 0)
        for p in pair_selected
    ]) == 1
)


# ============================================================
# For every possible pair:
#
# selected:
#       S(u) = +4 or -4
#
# not selected:
#       S(u) = 0
#
# and the symmetric shift has the same value.
# ============================================================

for u in range(1, L // 2):

    p = pair_selected[u - 1]

    solver.add(
        z3.Implies(
            p,
            z3.Or(
                S[u] == 4,
                S[u] == -4
            )
        )
    )

    solver.add(
        z3.Implies(
            z3.Not(p),
            S[u] == 0
        )
    )

    # Explicitly enforce the symmetry.
    solver.add(
        S[L - u] == S[u]
    )


# ============================================================
# u = L/2 cannot be nonzero.
#
# Otherwise there would only be ONE nonzero shift,
# because L/2 = L-(L/2).
# ============================================================

solver.add(
    S[L // 2] == 0
)


# ============================================================
# Solve
# ============================================================

print("=" * 60)
print(f"Solving L = {L}")
print("=" * 60)

start_time = time.time()

result = solver.check()

elapsed = time.time() - start_time

print("Z3 version:", z3.get_version_string())
print("Result:", result)
print(f"Time: {elapsed:.3f} seconds")


# ============================================================
# If SAT, extract solution
# ============================================================

if result == z3.sat:

    model = solver.model()

    A = [
        1 if z3.is_true(model.evaluate(a[i])) else 0
        for i in range(L)
    ]

    B = [
        1 if z3.is_true(model.evaluate(b[i])) else 0
        for i in range(L)
    ]

    print()
    print("A =", A)
    print("B =", B)

    print()
    print("Nonzero S(u):")

    for u in range(1, L):
        value = model.evaluate(S[u]).as_long()

        if value != 0:
            print(f"S({u}) = {value}")


# ============================================================
# UNKNOWN
# ============================================================

elif result == z3.unknown:

    print()
    print("Z3 returned UNKNOWN.")
    print("Reason:", solver.reason_unknown())


# ============================================================
# UNSAT
# ============================================================

else:

    print()
    print("No solution found for L =", L)