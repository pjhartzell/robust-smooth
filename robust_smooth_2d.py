"""
Adapted from Damian Garcia's MATLAB code:
- See Garcia, D., 2010, 'Robust smoothing of gridded data in one and higher 
dimensions with missing values in Computational Statistics and Data Analysis', 
Computational Statistics and Data Analysis.
- See Garcia, D., 2011, 'A fast all-in-one method for automated post-
processing of PIV data', Experiments in Fluids.

Copyright (c) 2017, Damien Garcia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of CREATIS, Lyon, France nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from scipy import ndimage
from scipy.fftpack import dct, idct
from scipy.optimize import fminbound


def robust_smooth_2d(y, **kwargs):
    """
    Interpolate missing values and smooth a 2D numpy array with optional robust
    outlier removal. Missing values in the array (where interpolation is
    desired) must be assigned numpy.nan values.

    Argument:
        y (numpy array): The 2D numpy array to be interpolated and smoothed.
    
    Optional Keyword Arguments:
        s (float): Smoothing factor to over-ride the automatically computed
            smoothing factor.
        robust (boolean): Apply robust outlier removal. Specify as True 
            (default) or False.
    
    Returns:
        z (numpy array): Smoothed version of the input array.
        s (float): Smoothing factor used to generate the output array.

    Examples:
        1. Allow automatic smoothing factor computation and robust outlier
           removal:
                robust_smooth_2d(y)
        2. Manually over-ride the smoothing factor computation:
                robust_smooth_2d(y, s=15)
        3. Over-ride smoothing factor computation and turn off the default
           robust smoothing:
                robust_smooth_2d(y, s=15, robust=False)
    """

    if "s" in kwargs:
        s = kwargs.get("s")
        auto_s = False
    else:
        auto_s = True

    if "robust" in kwargs:
        robust = kwargs.get("robust")
    else:
        robust = True

    size_y = np.asarray(y.shape)
    num_elements = np.prod(size_y)    
    not_finite = np.isnan(y)
    is_finite = np.logical_not(not_finite)
    num_finite = np.sum(is_finite)

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
    # and lower bounds for h are given to avoid under- or over-smoothing. 
    tensor_rank = sum(size_y!=1) # tensor rank of the y-array
    h_min = 1e-6
    h_max = 0.99
    s_min_bound = (((1 + np.sqrt(1 + 8*h_max**(2/tensor_rank)))/4/h_max**(2/tensor_rank))**2 - 1) / 16
    s_max_bound = (((1 + np.sqrt(1 + 8*h_min**(2/tensor_rank)))/4/h_min**(2/tensor_rank))**2 - 1) / 16

    # initialize stuff before iterating
    weights = np.ones(size_y)
    weights[not_finite] = 0
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
        while tolerance > 1e-5 and num_iterations < 100:
            num_iterations += 1
            dct_y = dct(dct(weights_total*(y - z) + z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
            
            # The generalized cross-validation (GCV) method is used to compute
            # the smoothing parameter S. Because this process is time-consuming,
            # it is performed from time to time (when the number of iterations
            # is a power of 2).
            if auto_s and not np.log2(num_iterations) % 1:
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

        if robust:
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
        else:
            robust_iterate = False

    return z, s


def robust_weights(y, z, is_finite, h):
    """Generate bi-square weights for robust smoothing (outlier rejection)."""
    residuals = y - z
    median_abs_deviation = np.median(np.fabs(residuals[is_finite] - np.median(residuals[is_finite])))
    studentized_residuals = np.abs(residuals/(1.4826*median_abs_deviation)/np.sqrt(1-h))
    # the weighting can be tuned by modifying the 4.685 value (make it smaller
    # for more aggressive outlier detection)
    bisquare_weights = ((1 - (studentized_residuals/4.685)**2)**2) * ((studentized_residuals/4.685) < 1)
    bisquare_weights[np.isnan(bisquare_weights)] = 0
    return bisquare_weights


def gcv(p, lmbda, dct_y, weights_total, is_finite, y, num_finite, num_elements):
    """Generalized Cross Validation for determining the smoothing factor."""
    s = 10**p
    Gamma = 1/(1+s*lmbda**2)
    y_hat = idct(idct(Gamma*dct_y, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    rss = np.linalg.norm(np.sqrt(weights_total[is_finite])*(y[is_finite]-y_hat[is_finite]))**2
    trace_H = np.sum(Gamma)
    gcv_score = rss / num_finite / (1-trace_H/num_elements)**2
    return gcv_score


def initial_guess(y, not_finite):
    """Generate an initial estimate of the smooth surface with missing values
    interpolated.
    """
    # Nearest neighbor interpolation of missing values. This can leave visible 
    # artifacts resulting from the nearest neighbor interpolation that is used.
    if not_finite.any():
        indices = ndimage.distance_transform_edt(not_finite, return_indices=True)[1]
        z = y[indices[0], indices[1]]
    else:
        z = y
    # coarse smoothing using a fraction of the DCT coefficients
    z = dct(dct(z, norm='ortho', type=2, axis=0), norm='ortho', type=2, axis=1)
    zero_start = np.ceil(np.array(z.shape)/10).astype(int)
    z[zero_start[0]:,:] = 0
    z[:,zero_start[1]:] = 0
    z = idct(idct(z, norm='ortho', type=2, axis=1), norm='ortho', type=2, axis=0)
    return z

