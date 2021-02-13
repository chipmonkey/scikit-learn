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

print("------------")

r3 = ti.query_expand(np.asarray(querypoint), 3)
print(f"query_expand: {r3}")