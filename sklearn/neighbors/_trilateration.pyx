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
from libc.math cimport fabs

from scipy.spatial.distance import cdist

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
    cdef np.ndarray r0sortorder
    cdef np.ndarray ref_dists

    cdef readonly DTYPE_t[:, ::1] data
    cdef readonly DTYPE_t[:, ::1] ref_points
    cdef public ITYPE_t[::1] idx_array
    cdef readonly DTYPE_t[:, ::1] distances

    cdef DistanceMetric dist_metric

    def __cinit__(self):
        self.data_arr = np.empty((1, 1), dtype=DTYPE, order='C')

    def __init__(self, data, ref=None, metric='euclidean', **kwargs):
        self.data_arr = check_array(data, dtype=DTYPE, order='C')

        # Ref points can be passed in
        # Otherwise they are generated based on input data
        if ref is None:
            ref = self._create_ref_points()
            print("generated synthetic reference points:")
            print(ref)

        self.ref_points_arr = check_array(ref, dtype=DTYPE, order='C')

        if self.data_arr.shape[1] != self.ref_points_arr.shape[1]:
            raise ValueError("Data and reference points must share same dimension")

        n_samples = self.data_arr.shape[0]
        n_features = self.data_arr.shape[1]

        self.dist_metric = DistanceMetric.get_metric(metric, **kwargs)
        # self.euclidean = (self.dist_metric.__class__.__name__
        #                   == 'EuclideanDistance')
        metric = self.dist_metric.__class__.__name__
        if metric not in VALID_METRICS:
            raise ValueError('metric {metric} is not valid for '
                             '{TrilaterationIndex}'.format(metric=metric,
                                                   **DOC_DICT))
        self.dist_metric._validate_data(self.data_arr)

        self.ref_dists = self.dist_metric.pairwise(self.data_arr, self.ref_points_arr)

        # allocate arrays for storage
        self.idx_array_arr = np.arange(n_samples, dtype=ITYPE)


        # Sort data by distance to refpoint[0]
        # TODO: See _simultaneous sort to see if we can speed this up:
        # (Right now since self.distances is 2D, we can't use this)
        # _simultaneous_sort(&self.distances,
        #                    &self.idx_array,
        #                    self.distances.shape[0])

        r0sortorder = np.argsort(self.ref_dists[:,0])
        self.ref_dists = self.ref_dists[r0sortorder]
        self.idx_array_arr = self.idx_array_arr[r0sortorder]

        self._update_memviews()
        # _find_nearest_sorted_2D(self.distances, -2)
        # _find_nearest_sorted_2D(self.distances, 62)
        # _find_nearest_sorted_2D(self.distances, 100)

    def _create_ref_points(self):
        """
        Create a set of well distributed reference points
        in the same number of dimensions as the input data

        For the love of all that is holy, make this better.
        """

        cdef DTYPE_t MAX_REF = 9999.0

        ndims = self.data_arr.shape[1]
        refs = [[0]*i+[MAX_REF]*(ndims-i) for i in range(ndims)]
        return np.asarray(refs)

    def _update_memviews(self):
        self.data = get_memview_DTYPE_2D(self.data_arr)
        self.idx_array = get_memview_ITYPE_1D(self.idx_array_arr)
        self.ref_points = get_memview_DTYPE_2D(self.ref_points_arr)
        self.distances = get_memview_DTYPE_2D(self.ref_dists)

    def query(self, X, k=5,
              return_distance=True,
              sort_results=True):
        """
        query the index for k nearest neighbors
        """

        if k != 1:
            raise ValueError("Right now this only works for k=1")

        print(f"X.shape: {X.shape}")

        if X.shape[0] == 1:
            results=self._query_one(X, k, return_distance, sort_results)
        else:
            results = [self._query_one(x, k=k,
                              return_distance=return_distance,
                              sort_results=sort_results) \
                       for x in X]

        return results

    def _query_one(self, X, k=5,
              return_distance=True,
              sort_results=True):
        """
        query the index for k nearest neighbors
        """
        if X.shape[0] != 1:
            raise ValueError("_query takes only a single X point"
                             "use query for multiple records")

        if X.shape[X.ndim - 1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")
        
        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        cdef NeighborsHeap heap = NeighborsHeap(X.shape[0], k)

        # Establish the distances from the query point to the reference points
        cdef np.ndarray q_dists
        q_dists = self.dist_metric.pairwise(X, self.ref_points_arr)

        # cdef DTYPE_t[:, ::1] Xarr
        # Xarr = get_memview_DTYPE_2D(np.asarray(X))
        # Xarr = np.asarray(X, dtype=DTYPE, order='C')
        cdef int best_idx, low_idx, low_idx_possible, high_idx, high_idx_possible
        cdef int test_idx
        cdef DTYPE_t best_dist, test_dist

        best_idx = _find_nearest_sorted_2D(self.distances, q_dists[0,0])
        # best_dist = self.dist_metric.dist(self.data[best_idx,:], &Xarr, X.shape[1])
        # print(self.data_arr[best_idx,:].shape)
        best_dist = cdist([self.data_arr[self.idx_array_arr[best_idx],:]], X)

        print(f"first best guess: {best_idx} data[{self.idx_array_arr[best_idx]}]")
        print(best_idx)
        print(np.asarray(self.data[best_idx,:]))

        print("starting best distance")
        print(best_dist)

        # Establish bounds between which to search
        low_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] - best_dist)
        high_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] + best_dist)
        low_idx = max(best_idx - 1, 0)
        high_idx = min(best_idx + 1, self.distances.shape[0])
        print(f"low_idx_possible: {low_idx_possible}; high_idx_possible: {high_idx_possible}")

        # Consider adding chunking here
        # So we test more than one new point at a time
        test_idx = best_idx

        while True:
            if low_idx <= low_idx_possible and high_idx >= high_idx_possible:
                print(f"breaking because {low_idx} <= {low_idx_possible} and {high_idx} >= {high_idx_possible}")
                break

            # Determine whether the next high or low point is a better test:
            lowdelta = fabs(self.distances[low_idx, 0] - q_dists[0, 0])
            highdelta = fabs(self.distances[high_idx, 0] - q_dists[0, 0])
            print(f"comparing: {low_idx}, {high_idx}")
            print(f"lowdelta: {lowdelta}, highdelta: {highdelta}")
            if lowdelta <= highdelta and low_idx >= low_idx_possible:
                test_idx = low_idx
                low_idx = low_idx - 1
            elif high_idx <= high_idx_possible:
                test_idx = high_idx
                high_idx = high_idx + 1
            elif low_idx >= low_idx_possible:
                test_idx = low_idx
                low_idx = low_idx - 1
            else:
                print("why?")
                break

            print(f"testing point at index {test_idx}: data[{self.idx_array[test_idx]}] {np.asarray(self.data[self.idx_array[test_idx],:])}")
            # Check that all pre-calculated distances are better than best
            sufficient = True
            for d, q in zip(self.distances[test_idx,1:], q_dists[0,1:]):
                print(f"testing that: {abs(d-q)} < {best_dist}")
                if abs(d-q) > best_dist:
                    sufficient = False
                    break
            
            if sufficient:
                test_dist = cdist([self.data_arr[self.idx_array[test_idx],:]], X)
                print(f"{test_idx} is sufficient... test_dist is: {test_dist}")
                if test_dist < best_dist:
                    print("replacing best with test...")
                    print(f"new best idx: {best_idx} ({self.idx_array[best_idx]}) with dist: {test_dist}")
                    best_idx = test_idx
                    best_dist = test_dist
                    low_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] - best_dist)
                    high_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] + best_dist)

            continue

        return (self.idx_array[best_idx], best_dist)



    def query_radius(self, X, r,
                     int return_distance=False,
                     int count_only=False,
                     int sort_results=False):
        """
        query the index for neighbors within a radius r
        """
        if count_only and return_distance:
            raise ValueError("count_only and return_distance "
                             "cannot both be true")

        if sort_results and not return_distance:
            raise ValueError("return_distance must be True "
                             "if sort_results is True")

    pass
