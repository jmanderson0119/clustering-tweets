import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import load_npz
from sklearn.metrics import silhouette_score

os.makedirs("../figures", exist_ok=True)

cos_sim = np.loadtxt("cosine_similarity.txt")
X = load_npz("tweet_features.npz").toarray()


# Converts to distance matrix for clustering and keeps only the upper triangle
dist_matrix = 1.0 - cos_sim
np.clip(dist_matrix, 0, None, out=dist_matrix)
condensed = squareform(dist_matrix, checks=False)

# Average link hierarchical clustering, returns a linkage matrix
dendro = linkage(condensed, method='average')
labels = fcluster(dendro, 49, criterion='maxclust')
np.savetxt("avg_link_labels.txt", labels, fmt="%d")

# Cluster size distribution
unique_clusters, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(8, 4))
plt.bar(unique_clusters, counts, color='purple', edgecolor='none')
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("../figures/avg_link_cluster_sizes.png", dpi=600)
plt.show()

# Cohesion, separation, silhouette
global_centroid = X.mean(axis=0)
sse, bss = 0.0, 0.0
for c in unique_clusters:
    members = X[labels == c]
    centroid = members.mean(axis=0)
    sse += np.sum((members - centroid) ** 2)
    bss += len(members) * np.sum((centroid - global_centroid) ** 2)

sil = silhouette_score(dist_matrix, labels, metric='precomputed')

print(f"\nSSE: {sse}")
print(f"BSS: {bss}")
print(f"Silhouette Score: {sil}")
