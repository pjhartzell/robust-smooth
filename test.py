
import numpy as np
import matplotlib.pyplot as plt
from robust_smooth_2d import robust_smooth_2d


def peaks(grid_size, num_peaks):
    xp = np.arange(grid_size)
    [x,y] = np.meshgrid(xp,xp)
    z = np.zeros_like(x).astype(float)
    for i in range(num_peaks):
        x0 = np.random.rand()*grid_size
        y0 = np.random.rand()*grid_size
        sdx = np.random.rand()*grid_size/4.
        sdy = sdx
        c = np.random.rand()*2 - 1.
        f = np.exp(-((x-x0)/sdx)**2-((y-y0)/sdy)**2 - (((x-x0)/sdx))*((y-y0)/sdy)*c)
        f *= np.random.rand()
        z += f
    return z


def add_noise(z, factor, fraction):
    std = np.std(z)
    num_elements = np.prod(z.shape)
    for i in range(np.floor(num_elements/fraction).astype(int)):
        row_idx = np.floor(np.random.rand()*z.shape[0]).astype(int)
        col_idx = np.floor(np.random.rand()*z.shape[1]).astype(int)
        z[row_idx,col_idx] += ((np.random.rand() - 0.5)*2) * std * factor
    return z


def missing_data(z, fraction):
    num_elements = np.prod(z.shape)
    for i in range(np.floor(num_elements/fraction).astype(int)):
        row_idx = np.floor(np.random.rand()*z.shape[0]).astype(int)
        col_idx = np.floor(np.random.rand()*z.shape[1]).astype(int)
        z[row_idx,col_idx] = np.nan
    return z


def bad_cluster(z, topleft, size, value):
    z[topleft[0]:topleft[0]+size, topleft[1]:topleft[1]+size] = value
    return z


# Create a 2d test array with some noise, missing values, and a bad cluster
np.random.seed(1)

y = peaks(256,50)
plt.imshow(y)
plt.title('Original')
plt.show()
original_clean = y.copy()

y = add_noise(y, 1.5, 5)
plt.imshow(y)
plt.title('Noisy')
plt.show()
original_noisy = y.copy()

y = missing_data(y, 10)
plt.imshow(y)
plt.title('Noisy + Missing')
plt.show()

y = bad_cluster(y, [50,50], 12, np.nan)
plt.imshow(y)
plt.title('Noisy + Missing + Bad Cluster')
plt.show()
original_dirty = y.copy()

z, s = robust_smooth_2d(y, robust=True)
print("s = {}".format(s))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=[10,5])
ax1.imshow(original_clean)
ax1.set_title('Clean')
ax2.imshow(original_noisy)
ax2.set_title('Noisy/Outliers')
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=[10,5])
ax1.imshow(original_dirty)
ax1.set_title('Noisy + Missing Data')
ax2.imshow(z)
ax2.set_title('Robust Smooth')
plt.show()
