""" Registration methods based on cross correlation """

from numpy import ndarray
import numpy as np

from thunder.rdds.images import Images
from thunder.imgprocessing.registration import RegistrationMethod
#from thunder.imgprocessing.regmethods.utils import computeDisplacement, computeReferenceMean, checkReference
from thunder.imgprocessing.transformation import GridTransformer, TRANSFORMATION_TYPES
from thunder.imgprocessing.regmethods.utils import volumesToMatrix, imageJacobian, solveLinearized, computeReferenceMean, checkReference


class LucasKanadeRegistration(RegistrationMethod):
    def __init__(self, maxIter=10, transformationType='Translation', tol=1e-3, robust=False, border=0):
#        self.jacobian_args = kwargs
        self.maxIter = maxIter
        self.transformationType = transformationType
        self.tol = tol
        self.robust = robust
        self.border = border

    def prepare(self, images, startidx=None, stopidx=None):
        """
        Prepare Lucas Kanade registration by computing or specifying a reference image.

        Parameters
        ----------
        images : ndarray or Images object
            Images to compute reference from, or a single image to set as reference

        See computeReferenceMean.
        """
        if isinstance(images, Images):
            self.reference = computeReferenceMean(images, startidx, stopidx)
        elif isinstance(images, ndarray):
            self.reference = images
        else:
            raise Exception('Must provide either an Images object or a reference')
        self.referenceMat = volumesToMatrix(self.reference)
        return self

    def isPrepared(self, images):
        """
        Check if Lucas-Kanade is prepared by checking the dimensions of the reference.

        See checkReference.
        """

        if not hasattr(self, 'reference'):
            raise Exception('Reference not defined')
        else:
            checkReference(self.reference, images)

    def getTransform(self, vol):
        # Create initial transformation
        tfm = TRANSFORMATION_TYPES[self.transformationType](shift=np.zeros(vol.ndim))
        grid = GridTransformer(vol.shape)
        iter = 0
        normDelta = np.inf
        while iter < self.maxIter and normDelta > self.tol:
            iter += 1
            volTfm, jacobian = imageJacobian(vol, tfm, grid, border=self.border)
            deltaTransformParams, coeff = solveLinearized(volumesToMatrix(volTfm), volumesToMatrix(jacobian), self.referenceMat, self.robust)
            tfm.updateParams(-deltaTransformParams)
            normDelta = np.linalg.norm(-deltaTransformParams)
        return tfm
