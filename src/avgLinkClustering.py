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

# Searches over a range of k values using silhouette as a guiding heuristic
# the optimum is actually 1531 and gives good cohesion and separation, and also finds
# meaningful clusterings, but its not very amenable to the heatmap visual nor the data itself. 
# This still shows the few standout clusters, but the performance is not as good
k_range = range(2, 100)
silhouette_scores = []

for k in k_range:
    labels_k = fcluster(dendro, k, criterion='maxclust')
    score = silhouette_score(dist_matrix, labels_k, metric='precomputed')
    silhouette_scores.append(score)
    print(f"k: {k} silhouette: {score}")

plt.figure(figsize=(8, 4))
plt.plot(list(k_range), silhouette_scores, marker='o', markersize=0.5, linewidth=0.5, color='purple')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("../figures/avg_link_silhouette_sweep.png", dpi=600)
plt.show()

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"Best: {best_k}")

labels = fcluster(dendro, best_k, criterion='maxclust')
np.savetxt("avg_link_labels.txt", labels, fmt="%d")

# Cluster size distribution
unique_clusters, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(8, 4))
plt.bar(unique_clusters, counts, color='purple', edgecolor='none')
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("../figures/avg_link_cluster_sizes.png", dpi=150)
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
