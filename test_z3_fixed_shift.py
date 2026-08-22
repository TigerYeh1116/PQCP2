import time
import z3


# ============================================================
# Parameters
# ============================================================

L = 44

# Each candidate gets this much time.
TIMEOUT_MS = 10_000


# ============================================================
# Periodic autocorrelation
# ============================================================

def periodic_autocorrelation(seq, shift):
    """
    rho(seq; shift)
        = L - 2 * sum(XOR(seq[i], seq[i+shift]))
    """

    return L - 2 * z3.Sum([
        z3.If(
            z3.Xor(
                seq[i],
                seq[(i + shift) % L]
            ),
            1,
            0
        )
        for i in range(L)
    ])


# ============================================================
# Solve for one fixed nonzero shift k
#
# Required:
#
#     S(k)   = +/-4
#     S(L-k) = +/-4
#
#     S(u) = 0 for all other u != 0
#
# We explicitly fix the sign as well.
# ============================================================

def solve_for_shift(k, sign):

    print()
    print("-" * 60)
    print(f"Testing k = {k}, sign = {sign}")
    print("-" * 60)

    solver = z3.Solver()
    solver.set(timeout=TIMEOUT_MS)

    # Binary variables
    a = [z3.Bool(f"a_{i}") for i in range(L)]
    b = [z3.Bool(f"b_{i}") for i in range(L)]

    # S(u) = rho(a;u) + rho(b;u)
    S = [
        periodic_autocorrelation(a, u)
        + periodic_autocorrelation(b, u)
        for u in range(L)
    ]

    # --------------------------------------------------------
    # u = 0
    # --------------------------------------------------------

    solver.add(S[0] == 2 * L)

    # --------------------------------------------------------
    # Symmetry
    #
    # S(u) = S(L-u)
    # --------------------------------------------------------

    for u in range(1, L):
        solver.add(
            S[L - u] == S[u]
        )

    # --------------------------------------------------------
    # Fixed nonzero pair
    # --------------------------------------------------------

    solver.add(
        S[k] == sign * 4
    )

    solver.add(
        S[L - k] == sign * 4
    )

    # --------------------------------------------------------
    # Everything else must be zero
    # --------------------------------------------------------

    for u in range(1, L):

        if u != k and u != L - k:
            solver.add(
                S[u] == 0
            )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    start = time.time()

    result = solver.check()

    elapsed = time.time() - start

    print("Result:", result)
    print(f"Time: {elapsed:.3f} seconds")

    # --------------------------------------------------------
    # SAT
    # --------------------------------------------------------

    if result == z3.sat:

        model = solver.model()

        A = [
            1 if z3.is_true(model.evaluate(x))
            else 0
            for x in a
        ]

        B = [
            1 if z3.is_true(model.evaluate(x))
            else 0
            for x in b
        ]

        print()
        print("FOUND SOLUTION")
        print("A =", A)
        print("B =", B)

        print()
        print("Nonzero autocorrelation sums:")

        for u in range(1, L):

            value = model.evaluate(S[u]).as_long()

            if value != 0:
                print(
                    f"S({u}) = {value}"
                )

        return A, B

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if result == z3.unknown:

        print(
            "Reason:",
            solver.reason_unknown()
        )

    return None


# ============================================================
# Main search
# ============================================================

print("=" * 60)
print(f"Fixed-shift Z3 search for L = {L}")
print(f"Timeout per case = {TIMEOUT_MS / 1000:.1f} seconds")
print("=" * 60)


# Because k and L-k are symmetric, only test:
#
#     k = 1 ... L/2 - 1
#
# For each k test +4 and -4.
#
# L = 44
# therefore:
#
#     k = 1 ... 21
#

for k in range(1, L // 2):

    for sign in (+1, -1):

        result = solve_for_shift(k, sign)

        if result is not None:

            print()
            print("=" * 60)
            print("SUCCESS")
            print(f"L = {L}")
            print(f"k = {k}")
            print(f"sign = {sign}")
            print("=" * 60)

            raise SystemExit


print()
print("=" * 60)
print("No solution found within the tested time limits.")
print("=" * 60)