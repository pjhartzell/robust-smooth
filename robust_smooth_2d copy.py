import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fftpack import dct, idct
from scipy.optimize import fminbound


def robust_smooth_2d(y):
    size_y = np.asarray(y.shape)
    num_elements = np.prod(size_y)    
    not_finite = np.isnan(y)
    is_finite = np.logical_not(not_finite)
    num_finite = np.sum(is_finite)

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
    tensor_rank = sum(size_y!=1) # tensor rank of the y-array
    h_min = 1e-6
    h_max = 0.99
    s_min_bound = (((1 + np.sqrt(1 + 8*h_max**(2/tensor_rank)))/4/h_max**(2/tensor_rank))**2 - 1) / 16
    s_max_bound = (((1 + np.sqrt(1 + 8*h_min**(2/tensor_rank)))/4/h_min**(2/tensor_rank))**2 - 1) / 16

    # initialize stuff before iterating
    weights_total = weights
    z = initial_guess(y, not_finite)
    z0 = z
    y[not_finite] = 0
    tolerance = 1
    num_robust_iterations = 1
    num_iterations = 0
    relaxation_factor = 1.75
    robust_iterate = True

    # iterative process
    while robust_iterate:
        amount_of_weights = np.sum(weights_total) / num_elements
        while tolerance > 1e-5 and num_iterations < 100:
            num_iterations += 1
            dct_y = dct(dct(weights_total*(y - z) + z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
            # The generalized cross-validation (GCV) method is used to compute
            # the smoothing parameter S. Because this process is time-consuming,
            # it is performed from time to time (when the number of iterations
            # is a power of 2).
            if not np.log2(num_iterations) % 2:
                p = fminbound(
                    gcv, 
                    np.log10(s_min_bound), 
                    np.log10(s_max_bound),
                    args=(lmbda, dct_y, weights_total, is_finite, y, num_finite, num_elements),
                    xtol=0.1,
                    full_output=False)

            s = 10**p
            Gamma = 1/(1+s*lmbda**2)
            z = relaxation_factor*idct(idct(Gamma*dct_y, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0) + (1-relaxation_factor)*z
            tolerance = np.linalg.norm(z0-z)/np.linalg.norm(z)
            z0 = z # re-initialize

        # average levereage
        h = 1
        for k in range(tensor_rank):
            h0 = np.sqrt(1 + 16*s)
            h0 = np.sqrt(1 + h0) / np.sqrt(2) / h0
        h = h*h0
        # take robust weights into account
        weights_total = weights*robust_weights(y, z, is_finite, h)
        # re-initialize for another iterative weighted process
        tolerance = 1
        num_iterations = 0
        num_robust_iterations += 1
        robust_iterate = num_robust_iterations < 4 # 3 robust iterations are enough
        print('here')

    return z, s


def robust_weights(y, z, is_finite, h):
    # weights for robust smoothing.
    residuals = y - z
    median_abs_deviation = np.median(np.fabs(residuals[is_finite] - np.median(residuals[is_finite])))
    studentized_residuals = np.abs(residuals/(1.4826*median_abs_deviation)/np.sqrt(1-h))
    bisquare_weights = ((1 - (studentized_residuals/4.685)**2)**2) * ((studentized_residuals/4.685) < 1)
    bisquare_weights[np.isnan(bisquare_weights)] = 0
    return bisquare_weights


def gcv(p, lmbda, dct_y, weights_total, is_finite, y, num_finite, num_elements):
    s = 10**p
    # print("s = {}".format(s))
    Gamma = 1/(1+s*lmbda**2)
    # print("Gamma[128,128] = {}".format(Gamma[128,128]))
    y_hat = idct(idct(Gamma*dct_y, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    rss = np.linalg.norm(np.sqrt(weights_total[is_finite])*(y[is_finite]-y_hat[is_finite]))**2
    trace_H = np.sum(Gamma)
    gcv_score = rss / num_finite / (1-trace_H/num_elements)**2
    return gcv_score


def initial_guess(y, not_finite):
    # nearest neighbor interpolation of missing values
    if not_finite.any():
        indices = ndimage.distance_transform_edt(not_finite, return_indices=True)[1]
        z = y[indices[0], indices[1]]
    else:
        z = y    
    # plt.imshow(z)
    # plt.show()
    # coarse smoothing using one-tenth (?) of DCT coefficients
    z = dct(dct(z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
    zero_start = np.ceil(np.array(z.shape)/10).astype(int)
    z[zero_start[0]:,:] = 0
    z[:,zero_start[1]:] = 0
    z = idct(idct(z, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    # plt.imshow(z)
    # plt.show()
    return z


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


def missing_data(z, hole_topleft, hole_size):
    num_elements = np.prod(z.shape)
    for i in range(np.floor(num_elements/2).astype(int)):
        row_idx = np.floor(np.random.rand()*z.shape[0]).astype(int)
        col_idx = np.floor(np.random.rand()*z.shape[1]).astype(int)
        z[row_idx,col_idx] = np.nan
    z[hole_topleft[0]:hole_topleft[0]+hole_size, hole_topleft[1]:hole_topleft[1]+hole_size] = np.nan
    return z

def add_noise(z):
    std = np.std(z)
    num_elements = np.prod(z.shape)
    for i in range(np.floor(num_elements/4).astype(int)):
        row_idx = np.floor(np.random.rand()*z.shape[0]).astype(int)
        col_idx = np.floor(np.random.rand()*z.shape[1]).astype(int)
        z[row_idx,col_idx] += ((np.random.rand() - 0.5)*2) * std * 0
    return z


# Create a 2d test array with some noise and missing values
np.random.seed(1)
y = peaks(256,50)
plt.imshow(y)
plt.show()
y = add_noise(y)
plt.imshow(y)
plt.show()
y = missing_data(y, [85,150], 50)
plt.imshow(y)
plt.show()

z, s = robust_smooth_2d(y)
print("s = {}".format(s))
plt.imshow(z)
plt.show()
