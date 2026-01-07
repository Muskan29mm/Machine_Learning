# Implementation of DBSCAN Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X,_ = make_blobs(n_samples=300, centers=4, random_state=42)

# scale data
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)


# Plot results
plt.scatter(X_scaled[:,0], X_scaled[:, 1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN clustering")
plt.show()

# eps → neighborhood radius

# min_samples → minimum points to form a cluster