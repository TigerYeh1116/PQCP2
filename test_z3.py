import z3

print("Z3 version:", z3.get_version_string())

x = z3.Int("x")

solver = z3.Solver()

solver.add(x > 5)
solver.add(x < 10)

result = solver.check()

print("Result:", result)

if result == z3.sat:
    print("x =", solver.model()[x])