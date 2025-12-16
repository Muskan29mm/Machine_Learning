# Example of Decision Tree Classifier on Spotify Dataset
# Classify if the genre is popular or not 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('spotify_2015_2025_85k.csv')

# Handle missing values
df['track_name'] = df['track_name'].fillna(df['track_name'].mode()[0])
df['album_name'] = df['album_name'].fillna(df['album_name'].mode()[0])

# ------------ IMPORTANT FIX -------------
# Label POPULAR vs NOT POPULAR using track-level popularity
df['popular_genre'] = np.where(df['popularity'] >= 50, 1, 0)
# ----------------------------------------

# Feature selection
X = df[['danceability', 'energy', 'key', 'loudness']]
y = df['popular_genre']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Decision Tree
plt.figure(figsize=(25, 12))
plot_tree(
    classifier,
    filled=True,
    feature_names=X.columns,
    class_names=['Not Popular', 'Popular'],
    fontsize=10
)
plt.show()
