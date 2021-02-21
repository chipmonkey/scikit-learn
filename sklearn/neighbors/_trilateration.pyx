#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

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

from collections import Counter
from functools import reduce
from libc.math cimport fabs, ceil

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
    cdef np.ndarray r_indexes
    cdef np.ndarray r_distances

    cdef readonly ITYPE_t[:, ::1] r_indexes_mv
    cdef readonly DTYPE_t[:, ::1] r_distances_mv

    cdef readonly DTYPE_t[:, ::1] data
    cdef readonly DTYPE_t[:, ::1] ref_points
    cdef public ITYPE_t[::1] idx_array
    cdef readonly DTYPE_t[:, ::1] distances

    cdef int ndims

    cdef DistanceMetric dist_metric


    def __cinit__(self):
        self.data_arr = np.empty((1, 1), dtype=DTYPE, order='C')

    def __init__(self, data, ref=None, metric='euclidean', **kwargs):
        self.data_arr = check_array(data, dtype=DTYPE, order='C')

        # Ref points can be passed in
        # Otherwise they are generated based on input data
        if ref is None:
            ref = self._create_ref_points()
            # print("generated synthetic reference points:")
            # print(ref)

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

        # Build ADDITIONAL indexes each sorted by the specific ref dists
        # Note that self.distances/self.ref_dists are ONLY sorted by r0 dist
        self.distances = get_memview_DTYPE_2D(self.ref_dists)
        self.data = get_memview_DTYPE_2D(self.data_arr)
        self.idx_array = get_memview_ITYPE_1D(self.idx_array_arr)
        self.ref_points = get_memview_DTYPE_2D(self.ref_points_arr)

        self._build_r_indexes()
        self.r_indexes_mv = get_memview_ITYPE_2D(self.r_indexes)
        self.r_distances_mv = get_memview_DTYPE_2D(self.r_distances)


    def _create_ref_points(self):
        """
        Create a set of well distributed reference points
        in the same number of dimensions as the input data

        For the love of all that is holy, make this better.
        """

        cdef DTYPE_t MAX_REF = 9999.0

        self.ndims = self.data_arr.shape[1]
        refs = [[0]*i+[MAX_REF]*(self.ndims-i) for i in range(self.ndims)]
        return np.asarray(refs)


    cdef _build_r_indexes(self):
        """
        Build a 2D array
        """
        self.r_indexes = np.zeros((self.distances.shape[1], self.distances.shape[0]), dtype=ITYPE, order='C')
        self.r_distances = np.zeros((self.distances.shape[1], self.distances.shape[0]), dtype=DTYPE, order='C')

        for i in range(self.distances.shape[1]):
            self.r_indexes[i,] = self.idx_array
            self.r_distances[i,] = self.distances[:,i]
            if i > 0:  # At this moment, the first column is already sorted
                sortorder = np.argsort(self.r_distances[i,])
                self.r_indexes[i,] = self.r_indexes[i,sortorder]
                self.r_distances[i,] = self.r_distances[i, sortorder]

        return 0

    def query(self, X, k=5,
              return_distance=True,
              sort_results=True):
        """
        query the index for k nearest neighbors
        """

        if isinstance(X, list):
            results = self._query_one(np.asarray(X), k, return_distance)
        elif X.shape[0] == 1:
            results=self._query_one(X, k, return_distance)
        else:
            raise NotImplementedError("Not yet")
            # [print(x) for x in X]
            # results = [self._query_one(np.asarray([x]), k=k,
            #                   return_distance=return_distance,
            #                   sort_results=sort_results) \
            #            for x in X]

        return results

    cdef _query_one(self, X, k=5,
              return_distance=True):
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

        # Can probably improve this - using a heap that allows more than 1 value
        # but we're always only using one here (X.shape[0] must be 1 from above)
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
        best_dist = cdist([self.data_arr[self.idx_array_arr[best_idx],:]], X)
        # heap.push(0, best_dist, best_idx)

        # Populate the heap using 2k elements; k on each side of our first guess:
        low_idx = max(best_idx - k, 0)
        high_idx = min(best_idx + k, self.distances.shape[0])
        for i in range(low_idx, high_idx + 1):
            if i < self.distances.shape[0]:
                test_dist = cdist([self.data_arr[self.idx_array_arr[i],:]], X)
                heap.push(0, test_dist, self.idx_array_arr[i])


        # Establish bounds between which to search
        if heap.largest(0) != np.inf:
            low_idx_possible = 0
            high_idx_possible = self.data_arr.shape[0]
        else:
            low_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] - heap.largest(0))
            high_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] + heap.largest(0))

        # Consider adding chunking here
        # So we test more than one new point at a time
        test_idx = best_idx

        while True:
            if low_idx <= low_idx_possible and high_idx >= high_idx_possible:
                break

            # Determine whether the next high or low point is a better test:
            lowdelta = fabs(self.distances[low_idx, 0] - q_dists[0, 0])
            highdelta = fabs(self.distances[high_idx, 0] - q_dists[0, 0])
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

            # print(f"testing point at index {test_idx}: data[{self.idx_array[test_idx]}] {np.asarray(self.data[self.idx_array[test_idx],:])}")
            # Check that all pre-calculated distances are better than best
            sufficient = True

            for d, q in zip(self.distances[test_idx,1:], q_dists[0,1:]):
                # print(f"testing that: {abs(d-q)} < {heap.largest(0)}")
                if abs(d-q) > heap.largest(0):
                    sufficient = False
                    break
            
            if sufficient:
                test_dist = cdist([self.data_arr[self.idx_array[test_idx],:]], X)
                # print(f"{test_idx} is sufficient... test_dist is: {test_dist}")
                if test_dist < heap.largest(0):
                    heap.push(0, test_dist, self.idx_array[test_idx])
                    # print(f"pushing idx: {best_idx} ({self.idx_array[best_idx]}) with dist: {test_dist}")
                    best_idx = test_idx
                    low_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] - heap.largest(0))
                    high_idx_possible = _find_nearest_sorted_2D(self.distances, q_dists[0, 0] + heap.largest(0))

            continue

        # return (self.idx_array[best_idx], best_dist)
        return heap.get_arrays()

    cdef _get_start_dist(self, X, k):
        """
        For the expanding radius approach, we must make an initial guess
        for a radius that may contain exactly k members.
        We do this by selecting the R0 reference distances, and simply
        expanding to k items, and returning the distance which separates
        the k points along that sole axis
        """
        cdef np.ndarray q_dists
        cdef ITYPE_t low_idx, high_idx

        q_dists = self.dist_metric.pairwise(X, self.ref_points_arr)
        
        ground_idx = _find_nearest_sorted_2D(self.distances, q_dists[0, 0])
        # ground_idx = np.searchsorted(self.distances, q_dists[0, 0], side="left")
        low_idx = ground_idx - (k//2)
        high_idx = ground_idx + ceil(k/2)

        return(self.distances[high_idx, 0] - self.distances[low_idx, 0])

    def query_expand(self, X, k, return_distance=True, sort_results=True):
        """
            X can be what...?
            A singlie list of one coordinate: [50, 25, 100]
        """
        cdef np.ndarray idx_results
        # cdef DTYPE_t[::1] dist_results
        cdef np.ndarray dist_results


        if isinstance(X, list):
            X = np.asarray(X)

        cdef NeighborsHeap heap = NeighborsHeap(X.shape[0], k)

        if X.shape[0] == 1:
            idx_results = np.asarray(self._query_expand(X, k))

        else:
            raise NotImplementedError("You can't do that yet")
            # [print(x) for x in X]
            # all_results = [self._query_expand(x, k) for x in X]
            # print(f"all_results: {all_results}")

        dist_results = self.dist_metric.pairwise(self.data_arr[idx_results], X)
        # best = np.argsort(dist_results)
        # dist_results = dist_results[best]
        # idx_results = idx_results[best]

        for idx, dist in zip(idx_results, dist_results):
            heap.push(0, dist, idx)

        if return_distance:
            return heap.get_arrays()

        return idx_results


    cdef _query_expand(self, X, k):
        """
        return k-nn by the expanding method...
        select a starting radius, iterate over query_radius
        while expanding and contracting the radius to
        arrive at a good result
        """

        if X.shape[0] != 1:
            raise ValueError("_query takes only a single X point"
                             "use query for multiple records")

        if X.shape[X.ndim - 1] != self.data.shape[1]:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        cdef DTYPE_t radius, too_high, too_low, new_radius, STD_SCALE, MAX_FUDGE
        cdef ITYPE_t approx_count, i, MAX_ITER
        cdef ITYPE_t[::1] approx_array


        MAX_ITER = 20
        MAX_FUDGE = 5.0
        STD_SCALE = 2.0

        radius = self._get_start_dist(X, k)

        too_low = 0
        too_high = np.inf
        # approx_array = self._query_radius_at_least_k(X, radius, k)
        approx_array = self._query_radius_r0_only(X, radius)
        
        approx_count = approx_array.shape[0]

        for i in range(MAX_ITER):
            if approx_count == k:
                break
            elif approx_count < k:
                too_low = radius
                new_radius = radius * STD_SCALE
                if new_radius > too_high:
                    new_radius = (too_high + radius) / STD_SCALE
                radius = new_radius
                approx_array = self._query_radius_at_least_k(X, radius, k)
                approx_count = approx_array.shape[0]
                continue
            elif approx_count > k * MAX_FUDGE:
                radius = (radius + too_low) / STD_SCALE
                approx_array = self._query_radius_at_least_k(X, radius, k)
                approx_count = approx_array.shape[0]
            else:
                continue
        
        return approx_array


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

        # cdef ITYPE_t[::1] idx_within_r = np.asarray([1])
        cdef np.ndarray idx_within_r = np.asarray([1])
        cdef np.ndarray dists_within_r

        if isinstance(X, list):
            X = np.asarray(X)

        if X.shape[0] == 1:
            idx_within_r = self._query_radius_approx(X, r)
        else:
            idx_within_r = [self._query_radius_approx(x, r) for x in X]

        dists_within_r = self.dist_metric.pairwise(self.data_arr[idx_within_r], X).reshape(idx_within_r.shape[0])
        valid_idx = [x <= r for x in dists_within_r]
        dists_within_r = dists_within_r[valid_idx]
        idx_within_r = idx_within_r[valid_idx]

        if sort_results:
            sort_order = np.argsort(dists_within_r)
            dists_within_r = dists_within_r[sort_order]
            idx_within_r = idx_within_r[sort_order]

        if return_distance:
            return idx_within_r, dists_within_r

        return idx_within_r


    def query_radius_approx(self, X, r):
        """
        query the index for neighbors within a radius r
        return approximate results only (by relying on index distances)
        """
        result = self._query_radius_approx(X, r)
        return result

    cdef _query_radius_approx(self, X, r):
        """

        """

        # cdef np.ndarray points_in_range = np.arange(self.r_indexes_mv.shape[0]).reshape(self.r_indexes_mv.shape[0])
        cdef np.ndarray q_dists
        # cdef ITYPE_t[::1] common_idx
        cdef np.ndarray common_idx
        cdef np.ndarray new_points

        point_counts = Counter()

        if isinstance(X, list) or X.ndim == 1:
            X = np.asarray([X])

        points_in_range = []
        q_dists = self.dist_metric.pairwise(X, self.ref_points_arr)
        # print(f"q_dists: {q_dists[0]} - {q_dists[0, 1]}")
        # print(f"r.indexes.shape: {self.r_indexes_mv.shape}")

        for i in range(self.ndims):
            # low_idx = _find_nearest_sorted_1D(self.r_distances_mv[i, :], q_dists[0, i] - r)
            low_idx = np.searchsorted(self.r_distances_mv[i, :], q_dists[0, i] - r, side="left")
            # high_idx = _find_nearest_sorted_1D(self.r_distances_mv[i, :], q_dists[0, i] + r)
            high_idx = np.searchsorted(self.r_distances_mv[i, :], q_dists[0, i] + r, side="right")

            # print(f"low_idx: {low_idx}, high_idx: {high_idx}")
            # print(f"valid IDS: {self.r_indexes[i, low_idx:high_idx]}")
            new_points = self.r_indexes[i, low_idx:high_idx]
            # print(f"new_points: {new_points} len: {len(new_points)}")
            # point_counts.update(new_points)  # Failed performance improvement test; leaving for posterity
            points_in_range.append(new_points)
        
        # THIS IS THE PERFORMANCE PROBLEM:
        common_idx = reduce(np.intersect1d, points_in_range)  # This works but seems slow (fastest so far tho)
        # common_idx = np.asarray([k for (k, v) in point_counts.items() if v == self.r_indexes_mv.shape[0]])

        # print(f"commonIDs: {common_idx}")
        return common_idx

    cdef _query_radius_at_least_k(self, X, r, k):
        """ for use with expanding estimates...
        break early if we've eliminated too many points
        """

        cdef np.ndarray q_dists
        cdef np.ndarray common_idx

        if isinstance(X, list) or X.ndim == 1:
            X = np.asarray([X])

        points_in_range = []
        new_points = []

        q_dists = self.dist_metric.pairwise(X, self.ref_points_arr)

        for i in range(self.r_indexes_mv.shape[0]):
            low_idx = np.searchsorted(self.r_distances_mv[i, :], q_dists[0, i] - r, side="left")
            high_idx = np.searchsorted(self.r_distances_mv[i, :], q_dists[0, i] + r, side="right")

            new_points = self.r_indexes[i, low_idx:high_idx]

            if points_in_range == []:
                points_in_range = new_points
            else:
                # THIS IS THE PERFORMANCE PROBLEM:
                points_in_range = np.intersect1d(points_in_range, new_points)  # This is the line that works, but slowly
            # At Least k voodoo:
            if len(points_in_range) < k:
                break
        
        common_idx = np.asarray(points_in_range)
        return common_idx


    def query_radius_t3(self, X, r):
        """
        query the index for neighbors within a radius r
        return approximate results only (by relying on index distances)
        """
        # result = np.asarray(self._query_radius_r0_only(X, r))
        result = self._query_radius_r0_only(X, r)

        return result

    cdef _query_radius_r0_only(self, X, r):
        """
        This approach uses only the original sort order from Ref Point 0
        and then brute forces whatever that index indicates MAY be in range

        NOT FINISHED
        """

        cdef DTYPE_t[:, ::1] q_dists # , dist_results
        cdef np.ndarray[DTYPE_t, ndim=1] dist_results
        # cdef DTYPE_t[::1] dist_results
        cdef ITYPE_t[::1] approx_idxs, results
        # cdef np.ndarray[DTYPE_t, ndim=2] q_dists, dist_results
        # cdef np.ndarray[ITYPE_t] approx_idxs #, points_in_range
        cdef ITYPE_t i, low_idx, high_idx, count

        if isinstance(X, list) or X.ndim == 1:
            X = np.asarray([X])

        # points_in_range = []

        q_dists = self.dist_metric.pairwise(X, self.ref_points_arr)
        # ground_idx = _find_nearest_sorted_2D(self.distances, q_dists[0, 0])

        # Which of the following two pairs are reliably faster?  Probably neither:
        # low_idx = np.searchsorted(self.r_distances_mv[0,:], q_dists[0, 0] - r, side="left")
        # high_idx = np.searchsorted(self.r_distances_mv[0,:], q_dists[0, 0] + r, side="right")

        low_idx = np.searchsorted(self.distances[:,0], q_dists[0, 0] - r, side="left")
        high_idx = np.searchsorted(self.distances[:,0], q_dists[0, 0] + r, side="right")
        
        # print(f"self.distances.shape: {self.distances.shape}, low_idx: {low_idx}, high_idx: {high_idx}")
        # print(f"self.distances[low_idx, :]: {np.asarray(self.distances[low_idx,:])}")
        # print(f"self.distances[low_idx+1, :]: {np.asarray(self.distances[low_idx+1,:])}")
        # print(f"self.distances[low_idx+2, :]: {np.asarray(self.distances[low_idx+2,:])}")

        # print(f"q_dists[0, :]: {q_dists[0, :]}")
        # print(f"self.distances - qd: {abs(np.asarray(self.distances[low_idx,:] - q_dists[0, :]))} vs r: {r}")

        # The next two lines work but are 10 times slower than just brute alone (as-is)
        # approx_idxs = np.asarray([i for i in range(low_idx, high_idx+1) \
        #               if np.all(abs(np.asarray(self.distances[i,:]) - q_dists[0, :]) <= r)])
        # approx_idxs = [self.idx_array[i] for i in approx_idxs]
        approx_idxs = self.idx_array[low_idx:high_idx+1]
        # print(f"approx_idxs (len {len(approx_idxs)}): {np.array(approx_idxs)}")

        dist_results = self.dist_metric.pairwise(self.data_arr[approx_idxs], X).reshape(approx_idxs.shape[0])
        # print(f"len(dist_results): {len(dist_results)} or {dist_results.shape[0]} low_idx:high_idx {low_idx}:{high_idx}")
        dist_results = dist_results.reshape(dist_results.shape[0])
        # print(f"dist_results: {np.asarray(dist_results)}")

        # cdef ITYPE_t result_count
        # result_count = np.count_nonzero(dist_results <= r)
        # print(f"dist_results <= r: {dist_results <= r}")
        # print(f"result_count: {result_count}")

        # result_arr = np.zeros(result_count, dtype=ITYPE)
        result_arr = np.zeros(len(dist_results), dtype=ITYPE)
        # results = get_memview_ITYPE_1D(result_arr)  # If you replace result_arr with results from here on you get nonsense.  I should understand, but do not.

        count = 0
        for i in range(len(dist_results)):
            # print(f"{i} (count: {count}) testing point {approx_idxs[i]} with dist: {dist_results[i]}")
            if dist_results[i] <= r:
                # if approx_idxs[i] == 18:
                #     print(f"adding {approx_idxs[i]}")
                result_arr[count] = approx_idxs[i]
                count += 1


        # NOW WHAT?
        # results_arr = approx_idxs[dist_results <= r]

        # Why is this slow: ?
        # results = approx_idxs[[range(8000)]]

        #  This alone is slow... why?
        # cdef np.array(dtype=bool) things  # Not sure how to declare this
        # cdef ITYPE_t[::1] things
        # things = [x <= r for x in dist_results]

        # for i in range(len(dist_results)):
        #     if things[i]:
        #         results.append(approx_idxs[i])

        # This is slow but works:
        # results = approx_idxs[[x[0] <= r for x in dist_results]]

        # This is even slower but works:
        # results = np.asarray([approx_idxs[i] for i in range(len(approx_idxs)) if dist_results[i] <= r])

        # print(f"results: {results}")
        # return results
        # return np.sort(np.asarray(results[0:count]))
        return  np.asarray(result_arr[0:count])


    cdef _query_radius_best_r_only(self, X, r):
        """
        This approach counts the candidates using each reference point by themselves
        then brute forces the indexees from the best possible result

        NOT FINISHED
        """
        pass