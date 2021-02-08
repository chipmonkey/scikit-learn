#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

# Trilateration (or n-fold lateration for arbitrary dimensions)
# =====================
#
#    Author: Chip Lynch <cmlync02@louisville.edu>
#    License: BSD something something
#
# The Trilateration Index itself is a only a data structure designed
# to store distances from n (default 3 thus "tri") fixed points in a
# sorted manner for quick approximate distance comparisons.  This is
# most beneficial when the distance metric is complex (i.e. accurate
# geographic distances on an oblate spheroid earth).
#
# Trilateration optimized nearest-neighbor and related algorithms will
# create this structure during the "fit" phase, and leverage the anchored
# distance data within neighbor searches.

cimport numpy as np
import numpy as np

from ..utils import (
    check_array
)

from ._typedefs cimport DTYPE_t, ITYPE_t, DITYPE_t
from ._typedefs import DTYPE, ITYPE

from ._dist_metrics cimport (DistanceMetric, euclidean_dist, euclidean_rdist,
                             euclidean_dist_to_rdist, euclidean_rdist_to_dist)


__all__ = ['TrilaterationIndex']

DOC_DICT = {'TrilaterationIndex': 'TrilaterationIndex'}

# Start simple:
VALID_METRICS = ['EuclideanDistance']

include "_helpers.pxi"

cdef class TrilaterationIndex:
    # __doc__ = CLASS_DOC.format(**DOC_DICT)
    
    cdef np.ndarray data_arr
    cdef np.ndarray ref_points_arr
    cdef np.ndarray idx_array_arr

    def __cinit__(self):
        self.data_arr = np.empty((1, 1), dtype=DTYPE, order='C')

    def __init__(self, data, metric='euclidean', **kwargs):
        self.data_arr = check_array(data, dtype=DTYPE, order='C')
        n_samples = self.data_arr.shape[0]
        n_features = self.data_arr.shape[1]

        self.dist_metric = DistanceMetric.get_metric(metric, **kwargs)
        self.euclidean = (self.dist_metric.__class__.__name__
                          == 'EuclideanDistance')
        metric = self.dist_metric.__class__.__name__
        if metric not in VALID_METRICS:
            raise ValueError('metric {metric} is not valid for '
                             '{TrilaterationIndex}'.format(metric=metric,
                                                   **DOC_DICT))
        self.dist_metric._validate_data(self.data_arr)

        # allocate arrays for storage
        self.idx_array_arr = np.arange(n_samples, dtype=ITYPE)

        self._update_memviews()

        print(self.data_arr)

    def _update_memviews(self):
        self.data = get_memview_DTYPE_2D(self.data_arr)
        self.idx_array = get_memview_ITYPE_1D(self.idx_array_arr)
        self.ref_points = get_memview_DTYPE_2D(self.ref_points_arr)

    def query(self, X, k=5,
              return_distance=True,
              sort_results=True):
        """
        query the index for k nearest neighbors
        """

        if X.shape[X.ndim - 1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")
        
        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")
        



    pass
