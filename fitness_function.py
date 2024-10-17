import numpy as np

def rastrigin(x):
    # Multimodal optimum 0
    n = len(x)
    z = 0
    for i in range(n):
        z += 10 + x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    return z

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    n = len(x)
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1))

def ackley(x):
    n = len(x)
    part1 = -0.2 * np.sqrt(1.0/n * sum(xi**2 for xi in x))
    part2 = -np.exp(1.0/n * sum(np.cos(2.0*np.pi*xi) for xi in x))
    return 20 + np.e + part1 + part2

def griewank(x):
    n = len(x)
    part1 = 1.0/4000 * sum(xi**2 for xi in x)
    part2 = np.prod(np.cos(xi/np.sqrt(i+1)) for i, xi in enumerate(x))
    return 1 + part1 - part2

def schwefel(x):
    return 418 * len(x) - sum(xi * np.sin(np.sqrt(np.abs(xi))) for xi in x)

def easom(x):
    return -np.cos(np.sqrt(sum(xi**2 for xi in x))) * np.exp(-sum(xi**2 for xi in x))

def penalty1(x):
    return sum((100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2) for i in range(len(x) - 1))

def levy_n13(x):
    if len(x) < 2:
        raise ValueError("Levy N. 13 function requires at least two dimensions")
    term1 = sum((i**2 * (x[i] - 1)**2 for i, xi in enumerate(x[:-1])))
    term2 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    return term1 + term2