# File to debug
# This file should never be published...
# Purge it from git before merging

import csv
import sklearn
import numpy as np

from scipy.spatial.distance import cdist

# samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
querypoint = [[50, 65]]

samples = []
refpoints = []

with open('/home/chipmonkey/repos/TrilaterationIndex/data/point_sample_10.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        samples.append([row['x'], row['y']])

with open('/home/chipmonkey/repos/TrilaterationIndex/data/sample_ref_points.csv') as csvfile:
    reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)
        refpoints.append([row['x'], row['y']])

print(refpoints)

print(samples)
print("Actual distances:")
print(cdist(samples, querypoint))

from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors import TrilaterationIndex

tree = KDTree(samples)
tq = tree.query_radius(querypoint, r=20)
print("tree query results:")
print(tq)

neigh = NearestNeighbors(radius=10)
neigh.fit(samples)

print("finding neighbors within radius:")
rng = neigh.radius_neighbors([[44., 44.]])
print("distances:")
print(np.asarray(rng[0][0]))
print("indexes:")
print(list(np.asarray(rng[1][0])))
print("points:")
things = [samples[x] for x in list(np.asarray(rng[1][0]))]
print(things)
# [samples[x] for x in list(np.asarray(rng[1][0]))]
# print(samples[list(np.asarray(rng[1][0]))])

print(type(tree))
print(type(tree.data))
# dist, ind = tree.query
print(np.asarray(tree.node_data))  # Ohhh https://stackoverflow.com/questions/20978377/cython-convert-memory-view-to-numpy-array
print(tree.node_data[0])

ti = TrilaterationIndex(samples, refpoints)



print(np.asarray(ti.ref_points))
print(np.asarray(ti.idx_array))
print("distances:")
print(np.asarray(ti.distances))

print(ti.ref_points.shape[0])
print(ti.ref_points.shape[1])

result1 = ti.query_radius_approx(np.asarray(querypoint), 20)
print(f"query_radius_approx: {result1}")


result2 = ti.query_radius_approx(np.asarray(querypoint), 20)
print(f"query_radius: {result2}")

rq = ti.query_radius(np.asarray(querypoint), 20)
print(f"rq: {rq}")

print("------------")
print("single point tests")
r4 = ti.query(querypoint, 3)
ids, dists = r4
print(f"query: {r4}")
print(f"indexes: {ids}, dists: {dists}")

r3 = ti.query_expand(querypoint, 3)
ids, dists = r3
print(f"query_expand: {r3}")
print(f"indexes: {np.asarray(ids)}", dists)

print("Passed Single Point tests")
print("------------")
print("Starting multi point tests")

querypoints = np.asarray([np.asarray([50, 65]),
               np.asarray([55, 70]),
               np.asarray([45, 80])])

r4 = ti.query(querypoints, 3)
print(f"query results r4: {r4}")
print(f"r4.shape: {np.asarray(r4).shape}")

ids, dists = r4
print(f"indexes: {ids}, dists: {dists}")

r3 = ti.query_expand(querypoints, 3)
ids, dists = r3
print(f"query_expand: {r3}")
print(f"indexes: {np.asarray(ids)}", dists)

# [(array([[14.30117536, 15.82017495, 16.12663017]]), array([[0, 5, 3]])),
#  (array([[16.08773716, 16.85576594, 16.85576594]]), array([[5, 0, 0]])),
#  (array([[ 4.37084771, 23.74124599, 23.74124599]]), array([[5, 3, 3]]))]

# cython sklearn/neighbors/_trilateration.pyx -a
# cython sklearn/neighbors/_kd_tree.pyx -a
