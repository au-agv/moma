import numpy as np
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.scipy.interpolate import RegularGridInterpolator
from jax import jit, vmap, grad

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

x = np.array([1, 10])
y = np.array([1, 10])
values = np.array([[3, 5], [10, 8]])
interpolator = RegularGridInterpolator((x, y), values, method="linear", fill_value=None)

def interp(*inputs):
    return interpolator(inputs)

def grad_interp(inputs):
    gradients = grad(interp, argnums=np.arange(len(inputs)))(*inputs)
    return np.array(gradients)

def variance_interval(means, stds):
    grid_slopes = grad_interp(means) * stds
    
    #TODO handle case where grid_slopes/scale is 0
    scale = np.sqrt(grid_slopes @ grid_slopes)
    
    move = grid_slopes / scale * stds

    min_point = means - move
    max_point = means + move
    ax.plot(min_point[0], min_point[1], 'k.')
    ax.plot(max_point[0], max_point[1], 'k.')
    print("min_point\n", min_point)
    print("max_point\n", max_point)

    min_results = interp(*min_point)
    max_results = interp(*max_point)

    return min_results, max_results

means = (np.array(1.5), np.array(2.5))
stds = (np.array(0.1), np.array(0.3))

sample_point = np.random.normal(means, stds, (10000, 2))
sample_results = interp(*sample_point.T)
sample_mean = np.mean(sample_results)
sample_std = np.std(sample_results)
print("sample_mean\n", sample_mean)
print("sample_std\n", sample_std)

theta = np.linspace(0, 2*np.pi, 100)
x_circ = np.cos(theta)*stds[0] + means[0]
y_circ = np.sin(theta)*stds[1] + means[1]
ax.plot(x_circ, y_circ, 'r-') 

sc = ax.scatter(sample_point[:, 0], sample_point[:, 1], c=sample_results, marker='.')
plt.colorbar(sc, ax=ax)

results = interp(*means)
result_min, results_max = variance_interval(means, stds)
print("reults_min\n", result_min)
print("results\n", results)
print("results_max\n", results_max)

ax.set_aspect('equal', adjustable='box')
plt.draw()
plt.show()

