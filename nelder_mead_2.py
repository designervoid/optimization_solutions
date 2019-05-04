import numpy as np
from scipy.optimize import minimize


def rosen(X):
    """The Rosenbrock function"""
    return np.sqrt(((X[0]**2 - 0.001**2)**2 + ((X[0]/X[1])*0.001)**2))

x0 = np.array([1.3, 0.7, 0.8])

res = minimize(rosen, x0, method='nelder-mead',
    options={'xtol': 1e-8, 'disp': True})
print(res.x)