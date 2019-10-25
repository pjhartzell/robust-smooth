import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def smoothn(y):
    size_y = np.asarray(y.shape)

    # Create weight matrix with zeros at missing value locations
    weights = np.ones(size_y)
    weights[np.isnan(y)] = 0

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
    z = initial_guess(y)



    return 0


def initial_guess(y):
    # nearest neighbor interpolation of missing values
    nan_map = np.isnan(y)
    if nan_map.any():
        indices = ndimage.distance_transform_edt(nan_map, return_indices=True)[1]
        z = y[indices[0], indices[1]]
    else:
        z = y
    


    return z
    
    #     ny = numel(y);    
    #     %-- coarse fast smoothing using one-tenth of the DCT coefficients
    #     siz = size(z{1});
    #     z = cellfun(@(x) dctn(x),z,'UniformOutput',0);
    #     for k = 1:ndims(z{1})
    #         for i = 1:ny
    #             z{i}(ceil(siz(k)/10)+1:end,:) = 0;
    #             z{i} = reshape(z{i},circshift(siz,[0 1-k]));
    #             z{i} = shiftdim(z{i},1);
    #         end
    #     end
    #     z = cellfun(@(x) idctn(x),z,'UniformOutput',0);
    # end



# Create a 2d test array with some missing values
np.random.seed(1)
y = np.random.rand(5,5)
y[0,1] = np.nan
y[2:4,3] = np.nan

z = smoothn(y)
# plt.imshow(z)
# plt.show()
