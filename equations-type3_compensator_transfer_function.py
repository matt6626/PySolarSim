from sympy import symbols, Eq, solve, S, pprint

# Define the symbols
s, adc, fp, r1, r2, r3, c1, c2, c3, wp = symbols(
    "s adc fp r1 r2 r3 c1 c2 c3 wp", real=True
)  # , real=True, positive=True
# )

# Define the equation
denom = (
    s**3 * (r1 + r3 + adc * wp * r1 * r3 * r3)
    + s**2
    * (
        (
            (r1 + r3) * c2
            + c1 * c2 * r2 * (r1 + r3) * wp
            + c1 * r2
            + adc * wp * c2 * r1 * r3 * (c1 + c3)
            + adc * wp * c1 * c3 * r1 * r2
        )
        / (c1 * c2 * r2)
    )
    + s
    * (
        (c2 * (r1 + r3) * wp + 1 + wp * c1 * r2 + (adc * wp * r1) * (c1 + c3))
        / (c1 * c2 * r2)
    )
    + wp / (c1 * c2 * r2)
)

# Print the equation
# print(denom)

# Set the equation equal to zero and solve
eq = Eq(denom, 0)
roots = solve(eq, s)

pprint(roots)
