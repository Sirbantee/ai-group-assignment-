from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize data
X_scaled = StandardScaler().fit_transform(X)

# Reduce dimensions for visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# GMM
gmm = GaussianMixture(n_components=10, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Plot function
def plot_clusters(data, labels, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=15)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

# Visualizations
plot_clusters(X_pca, kmeans_labels, "KMeans Clustering")
plot_clusters(X_pca, dbscan_labels, "DBSCAN Clustering")
plot_clusters(X_pca, gmm_labels, "GMM Clustering")

# Anomaly Detection Example (DBSCAN)
anomalies = np.where(dbscan_labels == -1)[0]
print(f"\nDetected {len(anomalies)} anomalies using DBSCAN.")

plot_clusters(X_pca, dbscan_labels, "Anomaly Detection (DBSCAN)")