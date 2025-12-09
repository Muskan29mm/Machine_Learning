# Implementation of K-means clustering

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
# print(data)

print(data[:5])

kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(data)

print("Centroids:\n", kmeans.cluster_centers_)
print("Labels:\n", kmeans.labels_)

plt.figure(figsize=(20,6))
plt.scatter(data[:,0], data[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', s=200, linewidths=3)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()

