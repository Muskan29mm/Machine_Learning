# Implementation of hierarchical clustering

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()

X = iris.data
Y = iris.target

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# Apply agglomerative clustering
agg_clust = AgglomerativeClustering(
    n_clusters=4,
    linkage = 'average'
)
data = agg_clust.fit_predict(X_scaled)
print(data)

# Evaluating
accuracy_result = accuracy_score(Y, data)
print(f"Model accuracy : {accuracy_result:.4f}")


# Compute the linkage matrix
linkage_matrix = linkage(X_scaled, method='average')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
dendrogram(linkage_matrix, color_threshold=1.35)
plt.axhline(y=1.35, color='black', linestyle='--')  # Example threshold line, change Y also to get correct cut visualization
plt.show()
