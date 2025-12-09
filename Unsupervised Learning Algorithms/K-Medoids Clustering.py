# Implementation of K-medoids clustering

from sklearn.datasets import load_iris
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data

kmedoids = KMedoids(n_clusters=5, random_state=5)
kmedoids.fit(data)

print("Centroids:\n", kmedoids.cluster_centers_)
print("Labels:\n", kmedoids.labels_)

plt.figure(figsize=(20,6))
plt.scatter(data[:,0], data[:,1], c=kmedoids.labels_)
plt.scatter(kmedoids.cluster_centers_[:,0], kmedoids.cluster_centers_[:,1], s=200, c='red', marker='x')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Medoids Clustering")
plt.show()

