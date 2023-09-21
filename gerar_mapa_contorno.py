import benchmark_functions as bf
import rps

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm

delta = 0.025
lu = [(-5, 5), (-5, 5)]
x = np.arange(lu[0][0], lu[0][1], delta)
y = np.arange(lu[1][0], lu[1][1], delta)
X, Y = np.meshgrid(x, y)
func = bf.rosenbrock_function
Z = [[func([X[i][j], Y[i][j]]) for j in range(len(X[0]))] for i in range(len(X))]
title = func.__name__

fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.gray, extent=(lu[0][0], lu[0][1], lu[1][0], lu[1][1]))
CS = ax.contour(X, Y, Z)
ax.set_title(title)
plt.savefig("mc.png")