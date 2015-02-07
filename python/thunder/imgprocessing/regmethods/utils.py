""" Shared utilities for registration methods """

import numpy as np
from numpy import ndarray

from thunder.rdds.images import Images


def computeReferenceMean(images, startidx, stopidx):
    """
    Compute a reference by taking the mean across images.

    Parameters
    ----------
    images : Images
            An Images object containg the image / volumes to compute reference from

    startidx : int, optional, default = None
        Starting index if computing a mean over a specified range

    stopidx : int, optional, default = None
        Stopping index if computing a mean over a specified range

    Returns
    -------
    refval : ndarray
        The reference image / volume
    """

    if not (isinstance(images, Images)):
        raise Exception('Input data must be Images or a subclass')

    if startidx is not None and stopidx is not None:
        range = lambda x: startidx <= x < stopidx
        n = stopidx - startidx
        ref = images.filterOnKeys(range)
    else:
        ref = images
        n = images.nimages

    reference = (ref.sum() / float(n)).astype(images.dtype)

    return reference


def checkReference(images, reference):
    """
    Check that a reference is an ndarray and matches the dimensions of images.

    Parameters
    ----------
    images : Images
        An Images object containing the image / volumes to check against the reference

    reference : ndarray
        A reference image / volume
    """

    if isinstance(reference, ndarray):
        if reference.shape != images.dims.count:
            raise Exception('Dimensions of reference %s do not match dimensions of data %s' %
                            (reference.shape, images.dims.count))
        else:
            raise Exception('Reference must be an array')


def computeDisplacement(arry1, arry2):
    """
    Compute an optimal displacement between two ndarrays.

    Finds the displacement between two ndimensional arrays. Arrays must be
    of the same size. Algorithm uses a cross correlation, computed efficiently
    through an n-dimensional fft.

    Parameters
    ----------
    arry1 : ndarray
        The first array

    arry2 : ndarray
        The second array
    """

    from numpy.fft import fftn, ifftn
    from numpy import unravel_index, argmax

    # get fourier transforms
    f2 = fftn(arry2)
    f1 = fftn(arry1)

    # get cross correlation
    c = abs(ifftn((f1 * f2.conjugate())))

    # find location of maximum
    maxinds = unravel_index(argmax(c), c.shape)

    # fix displacements that are greater than half the total size
    pairs = zip(maxinds, arry1.shape)
    adjusted = [d - n if d > n // 2 else d for (d, n) in pairs]

    return adjusted


def zeroBorder(vol, border=1, cval=0.0):
    """Zero-out boundary voxels of a volume.

    Parameters
    ----------
        vol: 3d volume
        border: scalar or tuple containing borders for each dimension
        cval: constant value to replace boundary voxels with, default is 0.0
    """
    vol = vol.copy() # create new copy so we don't overwrite vol
    dims = np.array(vol.shape)
    if np.size(border) == 1:
        border = border * np.ones(vol.ndim, dtype=int)
    border = np.array(border, dtype=int)
    # Don't apply border to singleton dimensions.
    border[dims == 1] = 0
    assert len(border) == vol.ndim
    if np.any(dims - border <= 0):
        raise ValueError('Border %s exceeds volume shape %s.' %
                (str(border), str(dims)) )
    for dim, bval in enumerate(border):
        if bval > 0:
            slices = [slice(bval) if d == dim else slice(None) for d in xrange(vol.ndim)]
            vol[slices] = cval
            slices[dim] = slice(-bval, None)
            vol[slices] = cval
    return vol


def imageGradients(im, sigma=None):
    """Compute gradients of volume in each dimension using a Sobel filter.

    Parameters
    ----------
    im : ndarray
        single volume
    sigma : float or tuple, optional, default = None
        smoothing amount to apply to volume before computing gradients
    """
    from scipy.ndimage.filters import gaussian_filter, sobel
    if sigma is not None:
       im = gaussian_filter(im, sigma)
    grads = [sobel(im, axis=dim, mode='constant') / 8.0 for dim in xrange(im.ndim)]
    return grads


def imageJacobian(vol, tfm, grid=None, sigma=None, normalize=True, border=1, order=1):
    """Compute Jacobian of volume w.r.t. transformation parameters

    Args:
        I: volume
        tfm: Transform object
        sigma: smoothing bandwidth for gradients (None for no smoothing)
        normalize: Whether to normalize images before aligning.
        border: Number or tuple of border sizes to zero after transforming.
    Returns:
        It: vectorized version of transformed volume
        J: nvoxels x nparameters Jacobian matrix
    """
    # Compute gradients
    # Transform image and gradients
    # Note: does not work if we compute gradients after transforming (borders?)
    #It = zeroBorder(tfm.apply(I), border)
    #TODO(ben): check that tfm then grad works
#        It = transform_vol(I, tfm, grid, order=order)
#        grads = [transform_vol(grad, tfm, grid, order=order) for grad in grads]
    if grid is None:
        grid = GridTransformer(vol.shape)
    grads = imageGradients(vol, sigma)
    It = tfm.apply(vol, grid, order=order)
    It = zeroBorder(It, border)
    grads = [tfm.apply(grad, grid, order=order) for grad in grads]
    grads = [zeroBorder(grad, border) for grad in grads]

    if normalize:
        normI = np.linalg.norm(It.ravel())
        if normI == 0.0:
          raise ValueError('Transform yields volume of zeroes.')
        grads = [(1. / normI) * I - (I * It).sum() / (normI**3) * It for I in grads]
        It /= normI

    jacobianVols = tfm.jacobian(grads, grid.homo_points)
    return It, jacobianVols


def solveLinearized(vec, jacobian, reference, robust=False):
    """Solve linearized registration problem for change in transformation parameters and weighting on reference basis.

    Parameters
    ----------
    vec : array, shape (nvoxels,)
        vectorized volume
    jacobian: array, shape (nvoxels, nparams)
        jacobian for each transformation parameter
    reference: array, shape (nvoxels, nbasis)
        array of vectorized reference volumes
    robust: bool, optional, default = False
        solve a least absolute deviation problem instead of least squares

    Returns
    -------
    deltaTransforms : array, shape (nparams,)
        optimal change in transformation parameters
    coeff : array, shape (nbasis,)
        optimal weighting of reference volumes
    """
    A = np.column_stack((jacobian, reference))
    if robust:
        from statsmodels.regression.quantile_regression import QuantReg
        quantile = 0.5
        model = QuantReg(vec, A).fit(q=quantile)
        params = model.params
    else:
        model = np.linalg.lstsq(A, vec)
        params = model[0]
    deltaTransform, coeff = np.split(params, [jacobian.shape[1]])
    return deltaTransform, coeff


def volumesToMatrix(vols):
    if not isinstance(vols, list):
        return v2v(vols)
    else:
        return np.column_stack([v2v(v) for v in vols]).squeeze()

def v2v(v, dims=None):
    if v.ndim == 1:
        assert dims
        assert v.size == np.prod(dims)
        return v.reshape(dims)
    else:
        return v.ravel()