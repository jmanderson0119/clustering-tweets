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

dist_matrix = 1.0 - cos_sim
np.clip(dist_matrix, 0, None, out=dist_matrix)
condensed = squareform(dist_matrix, checks=False)

dendro = linkage(condensed, method='average')

n_samples = dist_matrix.shape[0]
k_range = range(2, n_samples)
silhouette_scores = []

for k in k_range:
    labels_k = fcluster(dendro, k, criterion='maxclust')
    score = silhouette_score(dist_matrix, labels_k, metric='precomputed')
    silhouette_scores.append(score)
    if k % 100 == 0:
        print(f"k: {k} silhouette: {score}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"Best k: {best_k}")

plt.figure(figsize=(12, 4))
plt.plot(list(k_range), silhouette_scores, color='purple', linewidth=0.6)
plt.axvline(best_k, color='red', linestyle='--', linewidth=1.0, label=f'best k={best_k}')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/avg_link_silhouette.png", dpi=600)
plt.show()
