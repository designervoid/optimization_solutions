import numpy as np
from scipy.optimize import fmin_bfgs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
start_time = time.time()

Nfeval = 1


def rosen(X): # function
    return np.sqrt(((X[0]**2 - 0.001**2)**2 + ((X[0]/X[1])*0.001)**2))


def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], 10**6/rosen(Xi)))
    Nfeval += 1


print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' Y', ' X', ' Z', 'f(X)'))
x0 = np.array([221.1, 221.1, 221.1], dtype=np.double)
[xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
    fmin_bfgs(rosen,
              x0,
              callback=callbackF,
              maxiter=2000,
              full_output=True,
              retall=False)


# Настраиваем 3D график
fig = plt.figure(figsize=[15, 10])
ax = fig.gca(projection='3d')

# Задаем угол обзора
ax.view_init(45, 30)

# Создаем данные для графика
X = np.arange(-2, 2, 0.1)
Y = np.arange(-1, 3, 0.1)
X, Y = np.meshgrid(X, Y)
Z = rosen(np.array([X,Y]))

# Рисуем поверхность
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()
print("--- algorithm complete in %s seconds ---" % (time.time() - start_time))