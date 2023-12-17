import time
from scipy.optimize import fsolve
from function_timer import FunctionTimer


# Define the system of equations
def equations(vars):
    x, y = vars
    eq1 = x**2 + y - 2
    eq2 = y**2 + x - 2
    return [eq1, eq2]


# Define the Jacobian matrix
def jacobian(vars):
    x, y = vars
    return [[2 * x, 1], [1, 2 * y]]


# Initial guess
x_init = 1
y_init = 1
guess = [x_init, y_init]

timer = FunctionTimer(
    func1=lambda: fsolve(equations, guess),
    func2=lambda: fsolve(equations, guess, fprime=jacobian),
    timeout=5,
)
timer.time()
