import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fftpack import dct, idct


def robust_smooth_2d(y):
    size_y = np.asarray(y.shape)
    num_elements = np.prod(size_y)
    not_finite = np.isnan(y)

    # Create weight matrix with zeros at missing (NaN) value locations
    weights = np.ones(size_y)
    weights[not_finite] = 0

    # Create the Lambda tensor, which contains the eingenvalues of the 
    # difference matrix used in the penalized least squares process. We assume 
    # equal spacing in horizontal and vertical here.
    lmbda = np.zeros(size_y)
    for i in range(y.ndim):
        size_0 = np.ones((y.ndim,), dtype=int)
        size_0[i] = size_y[i]
        lmbda += 2 - 2*np.cos(np.pi*(np.reshape(np.arange(0,size_y[i]), size_0))/size_y[i])

    # Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter for m = 2 (Equation #12
    # in the referenced CSDA paper).
    N = sum(size_y!=1) # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    sMinBnd = (((1 + np.sqrt(1 + 8*hMax**(2/N)))/4/hMax**(2/N))**2 - 1) / 16
    sMaxBnd = (((1 + np.sqrt(1 + 8*hMin**(2/N)))/4/hMin**(2/N))**2 - 1) / 16

    # initialize stuff before iterating
    weights_total = weights
    z = initial_guess(y, not_finite)
    y[not_finite] = 0
    tolerance = 1
    robust_step = 1
    num_iterations = 0
    relaxation_factor = 1.75
    iterate = True

    # iterative process
    while iterate:
        amount_of_weights = np.sum(weights_total) / num_elements
        while tolerance > 1e-3 and num_iterations < 100:
            num_iterations += 1
            dct_y = dct(weights_total*(y - z) + z)
            if 



    return 0


def initial_guess(y, not_finite):
    # nearest neighbor interpolation of missing values
    if nan_map.any():
        indices = ndimage.distance_transform_edt(not_finite, return_indices=True)[1]
        z = y[indices[0], indices[1]]
    else:
        z = y    
    # coarse smoothing using one-tenth (?) of DCT coefficients
    z = dct(dct(z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
    zero_start = np.ceil(np.array(z.shape)/3).astype(int)
    z[zero_start[0]:,:] = 0
    z[:,zero_start[1]:] = 0
    z = idct(idct(z, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    return z



# Create a 2d test array with some missing values
np.random.seed(1)
y = np.random.rand(10,20)
y[0,1] = np.nan
y[2:4,3] = np.nan

z = smoothn(y)
# plt.imshow(z)
# plt.show()
