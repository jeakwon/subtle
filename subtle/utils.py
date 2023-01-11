
import copy
import pickle
import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def load(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def avatar_preprocess(X):
    T, VC = X.shape
    C = 3
    X = X.reshape(T, VC//C, C)
    X -= X.mean((0, 1))
    X = X.reshape(T, VC)
    return X

def getDensityBounds(density, thresh=1e-6):
    # https://github.com/bermanlabemory/motionmapperpy/tree/master/motionmapperpy
    """
    Get the outline for density maps.
    :param density: m by n density image.
    :param thresh: Density threshold for boundaries. Default 1e-6.
    :return: (p by 2) points outlining density map.
    """
    x_w, y_w = np.where(density > thresh)
    x, inv_inds = np.unique(x_w, return_inverse=True)
    bounds = np.zeros((x.shape[0] * 2 + 1, 2))
    for i in range(x.shape[0]):
        bounds[i, 0] = x[i]
        bounds[i, 1] = np.min(y_w[x_w == bounds[i, 0]])
        bounds[x.shape[0] + i, 0] = x[-i - 1]
        bounds[x.shape[0] + i, 1] = np.max(y_w[x_w == bounds[x.shape[0] + i, 0]])
    bounds[-1] = bounds[0]
    bounds[:, [0, 1]] = bounds[:, [1, 0]]
    return bounds.astype(int)
    
def findPointDensity(zValues, sigma, numPoints, rangeVals):
    # https://github.com/bermanlabemory/motionmapperpy/tree/master/motionmapperpy
    """
    findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
    through convolving with a gaussian function.
    :param zValues: 2d points of shape (m by 2).
    :param sigma: standard deviation of smoothing gaussian.
    :param numPoints: Output density map dimension (n x n).
    :param rangeVals: 1 x 2 array giving the extrema of the observed range
    :return:
        bounds -> Outline of the density map (k x 2).
        xx -> 1 x numPoints array giving the x and y axis evaluation points.
%       density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)
    [XX, YY] = np.meshgrid(xx, yy)
    G = np.exp(-0.5 * (np.square(XX) + np.square(YY)) / np.square(sigma))
    Z = np.histogramdd(zValues, bins=[xx, yy])[0]
    Z = Z / np.sum(Z)
    Z = np.pad(Z, ((0, 1), (0, 1)), mode='constant', constant_values=((0, 0), (0, 0)))
    density = fftshift(np.real(ifft2(np.multiply(fft2(G), fft2(Z))))).T
    density[density < 0] = 0
    bounds = getDensityBounds(density)
    return bounds, xx, density
