""" Transformations produced by registration methods """
import numpy as np
from thunder.imgprocessing.regmethods.utils import zeroBorder, imageGradients, v2v
from thunder.utils.decorators import serializable

class Transformation(object):
    """ Base class for transformations """

    def apply(self, im):
        raise NotImplementedError


class DifferentiableTransformation(Transformation):
    """
    Differentiable transformations must have methods to ompute the Jacobian and update parameters. These are used by
    iterative alignment techniques like Lucas-Kanade and RASL.
    """
    def jacobian(self, imageGradients, imageGrid):
        """Compute gradient of transformation output with respect to parameters.

        Parameters
        ----------
        imageGradients : list of volumes, one for each dimension
            gradient of volume in each dimension
        imageGrid : list of arrays, one for each dimension
            each array contains the coordinates along that dimension for each voxel

        Returns
        -------
        ordered list of arrays containing the jacobian for each parameter
        """
        raise NotImplementedError

    def updateParams(self, deltaParams):
        """Update each parameter in the transformation.

        Parameters
        ----------
        deltaParams : array, shape (nparams,)
            ordered changes in parameters to be applied. The ordering here will match
            the ordering of the parameters returned by jacobian.
        """
        raise NotImplementedError


class AffineTransformation(DifferentiableTransformation):
    """ Affine transformations are differentiable and can be represented as a matrix."""
    def matrix(self):
        raise NotImplementedError

    def apply(self, vol, grid=None, order=1):
        if grid is None:
            center = getattr(self, 'center', None)
            grid = GridTransformer(vol.shape, center=center)
        A = self.matrix()
        return grid.transform_vol(vol, A, order=order)


class TranslationTransformation(AffineTransformation):
    def __init__(self, shift):
        """Translation in 3d.

        Parameters
        ----------
            shift: 3d vector of translations (x, y, z)

        """
        self.delta = shift

    def jacobian(self, imageGradients, imageGrid):
        return imageGradients

    def matrix(self):
        return transformationMatrix(self.delta)

    def updateParams(self, deltaParams):
        self.delta += deltaParams

class EuclideanTransformation(AffineTransformation):
    def __init__(self, shift, rotation=None, zTranslation=False, zRotation=False):
        """Translation and rotation in 3d.

        Parameters
        ----------
        shift :
        rot :
        """
        self.shift = np.atleast_1d(shift)
        self.ndim = len(self.shift)
        if rotation is None:
            if self.ndim == 2 or not zRotation:
                rotation = 0.0
            else:
                rotation = np.zeros(3)
        self.rotation = np.atleast_1d(rotation)
        self.zRotation = zRotation
        self.zTranslation = zTranslation

    def updateParams(self, deltaParams):
        if self.ndim == 2:
            self.shift += deltaParams[:2]
            self.rotation += deltaParams[2]
        else:
            self.shift += deltaParams[:3]
            self.rotation += deltaParams[3:]

    def jacobian(self, imageGradients, imageGrid):
        ndim = len(self.shift)

        stheta = np.sin(self.rotation[0])
        ctheta = np.cos(self.rotation[0])

        if ndim == 2 or not self.zRotation:
            dtheta =   imageGradients[0] * (-imageGrid[0] * stheta  + imageGrid[1] * -ctheta)
            dtheta +=  imageGradients[1] * ( imageGrid[0] * ctheta  + imageGrid[1] * -stheta)
            dangles = [dtheta]
        else:
            sphi = np.sin(self.rotation[1])
            cphi = np.cos(self.rotation[1])
            spsi = np.sin(self.rotation[2])
            cpsi = np.cos(self.rotation[2])
            dtheta = imageGradients[0] * (
               -imageGrid[0] * cphi * stheta +
                imageGrid[1] * (-ctheta * cpsi - stheta * sphi * spsi) +
                imageGrid[2] * (-cpsi * stheta * sphi + ctheta * spsi))
            dtheta +=  imageGradients[1] * (
                imageGrid[0] * ctheta * cphi +
                imageGrid[1] * (-cpsi * stheta + ctheta * sphi * spsi) +
                imageGrid[2] * (ctheta * cpsi * sphi + stheta * spsi))
            dphi =  imageGradients[0] * (
                imageGrid[2] * ctheta * cphi * cpsi -
                imageGrid[0] * ctheta * sphi +
                imageGrid[1] * ctheta * cphi * spsi)
            dphi += imageGradients[1] * (
                imageGrid[2] * cphi * cpsi * stheta -
                imageGrid[0] * stheta * sphi +
                imageGrid[1] * cphi * stheta * spsi)
            dphi += imageGradients[2] * (
               -imageGrid[0] * cphi -
                imageGrid[2] * cpsi * sphi -
                imageGrid[1] * sphi * spsi)
            dpsi = imageGradients[0] * (
                imageGrid[1] * (ctheta * cpsi * sphi + stheta * spsi) +
                imageGrid[2] * (cpsi * stheta - ctheta * sphi * spsi) )
            dpsi += imageGradients[1] * (
                imageGrid[1] * (cpsi * stheta * sphi - ctheta * spsi) +
                imageGrid[2] * (-ctheta * cpsi - stheta * sphi * spsi))
            dpsi += imageGradients[2] * (imageGrid[1] * cphi * cpsi - imageGrid[2] * cphi * spsi)
            dangles = [dtheta, dphi, dpsi]

        # Zero-out Jacobian corresponding to z transltation
        if ndim == 3 and not self.zTranslation:
            imageGradients[2][:] = 0.0
        return imageGradients + dangles

    def matrix(self):
        return transformationMatrix(self.shift,  self.rotation)

    def __repr__(self):
        return "EuclideanTransformation(shift=%s, rotation=%s)" % (repr(self.shift), repr(self.rotation))

class Displacement(Transformation):
    """
    Class for transformations based on spatial displacements.

    Can be applied to either images or volumes.

    Parameters
    ----------
    delta : list
        A list of spatial displacements for each dimensino,
        e.g. [10,5,2] for a displacement of 10 in x, 5 in y, 2 in z
    """

    def __init__(self, delta=None):
        self.delta = delta

    def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        return shift(im, map(lambda x: -x, self.delta), mode='nearest')

    def __repr__(self):
        return "Displacement(delta=%s)" % repr(self.delta)


class PlanarDisplacement(Transformation):
    """
    Class for transformations based on two-dimensional spatial displacements.

    Applied separately to each plane of a three-dimensional volume.

    Parameters
    ----------
    delta : list
        A nested list, where the first list is over planes, and
        for each plane a list of [x,y] displacements
    """

    def __init__(self, delta=None):
        self.delta = delta

    def apply(self, im):
        """
        Apply an 2D displacement by shifting each plane of a volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        if im.ndim == 2:
            return shift(im,  map(lambda x: -x, self.delta[0]))
        else:
            im.setflags(write=True)
            for z in range(0, im.shape[2]):
                im[:, :, z] = shift(im[:, :, z],  map(lambda x: -x, self.delta[z]), mode='nearest')
            return im

    def __repr__(self):
        return "PlanarDisplacement(delta=%s)" % repr(self.delta)


class GridTransformer(object):
    def __init__(self, dims, center=None):
        """Class to represent and transform a fixed grid of points over indices into a volume.

        Args:
            dims: vol_shape[::-1]
            center: center location of grid, defaults to the centroid.
                    Affine transfroms will be applied to the center-subtracted grid,
                    so the center corresponds to the point of rotation.
        """
        self.dims = np.array(dims)
        self.ndim = len(dims)
        if center is None:
            center = (self.dims - 1) / 2.0
        self.center = center
        self.homo_points = np.ones((self.ndim + 1,) + dims)
        self.homo_points[:-1, ...] = np.array(np.mgrid[[slice(p) for p in self.dims]])
        self.raw_points = self.homo_points.copy()
        self.world_to_index_tfm = transformationMatrix(self.center, np.zeros(self.ndim))
        self.index_to_world_tfm = transformationMatrix(-self.center, np.zeros(self.ndim))
        self.homo_points = self.transform_grid_world(self.index_to_world_tfm)
        self.index_points = self.transform_grid_world(self.world_to_index_tfm)

    def transform_grid_world(self, A):
        """Get the grid of points in world space after applying the given affine transform."""
        return np.tensordot(A.T, self.homo_points, axes=(0,0))

    def transform_grid(self, A):
        """Get the grid of points in index space after applying the given affine transform
           in world space.

        Args:
            A: 4x4 Affine transformation matrix
        Returns:
            y: 4 x dims[0] x dims[1] x dims[2] matrix containing the grid
        """
        return self.transform_grid_world(np.dot(self.world_to_index_tfm, A))

    def transform_vol(self, vol, A, **kwargs):
        from scipy.ndimage import map_coordinates
        new_grid = self.transform_grid(A)[:-1]
        transformed_vol = map_coordinates(vol, new_grid, cval=0.0, **kwargs)
        return transformed_vol


def transformationMatrix(shift, rot=None):
    """Create an affine transformation matrix

    Parameters
    ----------
    shift : array, shape (ndim,)
        translations along each dimension
    rot : scalar or array with shape (ndim, )

    Returns
    -------
    A : array, shape (ndims + 1, ndims + 1)
        transformation matrix that shifts and rotates a set of points in homogenous coordinates
    """

    ndim = len(shift)
    if rot is None:
        rot = np.zeros(ndim)
    else:
        rot = np.atleast_1d(rot)
    c = np.cos(rot)
    s = np.sin(rot)
    trans = np.eye(ndim + 1)
    trans[:-1, -1] = shift
    xrot = np.array(
            [[c[0], -s[0],  0,    0],
            [s[0],  c[0],  0,    0],
            [0,     0,     1,    0],
            [0,     0,     0,    1]])
    if ndim == 2 or np.size(rot) == 1:
        A = np.dot(trans, xrot[:ndim + 1, :ndim + 1])
    else:
        yrot = [[c[1],  0,     s[1], 0,],
                [0,     1,     0,    0],
                [-s[1], 0,     c[1], 0],
                [0,     0,     0,    1]]
        zrot = [[1,     0,     0,    0],
                [0,     c[2], -s[2], 0],
                [0,     s[2],  c[2], 0],
                [0,     0,     0,    1]]
        A = np.dot(trans, np.dot(xrot, np.dot(yrot, zrot)))
    return A

TRANSFORMATION_TYPES = {
    'Translation' : TranslationTransformation,
    'Euclidean' : EuclideanTransformation
}