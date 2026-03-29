import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics import silhouette_score
from collections import deque

os.makedirs("../figures", exist_ok=True)

jaccard_sim = np.loadtxt("jaccard_similarity.txt")
X = load_npz("tweet_features.npz").toarray()
n = jaccard_sim.shape[0]
dist = 1.0 - jaccard_sim

def build_snn_matrix(sim_matrix, k):
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -1  # excludes self
        top_k = np.argsort(row)[::-1][:k]
        A[i, top_k] = 1.0
    snn = (A @ A.T).astype(np.float32)
    np.fill_diagonal(snn, 0)
    return snn

def run_snn_clustering(snn_sim, eps, min_pts):
    labels = np.full(n, -1, dtype=int)
    core_points = set(np.where((snn_sim >= eps).sum(axis=1) >= min_pts)[0])

    cluster_id = 0
    visited = set()

    for core_point in core_points:
        if core_point in visited:
            continue
        queue = deque([core_point])
        visited.add(core_point)
        while queue:
            p = queue.popleft()
            labels[p] = cluster_id
            if p in core_points:
                neighbors = np.where(snn_sim[p] >= eps)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        cluster_id += 1

    return labels

# Joint search over the hyperparameters
k_values = list(range(2, 16))
eps_values = list(range(2, 8))
min_pts_values = [2, 3, 4, 5]

best_score = -np.inf
best_params = None
best_labels = None

for k in k_values:
    snn_sim = build_snn_matrix(jaccard_sim, k)
    for eps in eps_values:
        for min_pts in min_pts_values:
            labels_try = run_snn_clustering(snn_sim, eps, min_pts)
            n_clusters = len(set(labels_try[labels_try != -1]))
            n_noise = int((labels_try == -1).sum())
            valid = labels_try != -1

            if n_clusters < 2 or valid.sum() < 2:
                print(f"k: {k} eps: {eps} min_pts: {min_pts}   clusters: {n_clusters} skipped")
                continue

            sil = silhouette_score(dist[np.ix_(valid, valid)], labels_try[valid], metric='precomputed')
            coverage = valid.sum() / n
            composite = sil * coverage

            print(f"k: {k} eps: {eps}, min_pts: {min_pts}   clusters: {n_clusters} noise: {n_noise} sil: {sil} coverage={coverage} composite: {composite}")

            if composite > best_score:
                best_score = composite
                best_params = (k, eps, min_pts)
                best_labels = labels_try.copy()

best_k, best_eps, best_min_pts = best_params
print(f"\nOptimal params\n k: {best_k} eps: {best_eps} min_pts: {best_min_pts}")
np.savetxt("snn_labels.txt", best_labels, fmt="%d")

unique_clusters = np.unique(best_labels[best_labels != -1])
counts = [(best_labels == c).sum() for c in unique_clusters]
n_noise = int((best_labels == -1).sum())

print(f"clusters: {len(unique_clusters)}, noise: {n_noise} ({100 * n_noise / n:.1f}%)")
for c, cnt in zip(unique_clusters, counts):
    print(f"cluster {c}: {cnt} tweets")

plt.figure(figsize=(8, 4))
plt.bar(unique_clusters, counts, width=0.5, color='darkgreen', edgecolor='none')
plt.xticks([])
plt.title(f"Cluster Size Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("../figures/snn_cluster_sizes.png", dpi=600)
plt.show()

# Cohesion, separation, silhouette
valid_mask = best_labels != -1
X_valid = X[valid_mask]
labels_valid = best_labels[valid_mask]
global_centroid = X_valid.mean(axis=0)

sse, bss = 0.0, 0.0
for c in unique_clusters:
    members = X_valid[labels_valid == c]
    centroid = members.mean(axis=0)
    sse += np.sum((members - centroid) ** 2)
    bss += len(members) * np.sum((centroid - global_centroid) ** 2)

sil = silhouette_score(dist[np.ix_(valid_mask, valid_mask)], labels_valid, metric='precomputed')

print(f"\nSSE: {sse}")
print(f"BSS: {bss}")
print(f"Silhouette Score: {sil}")
