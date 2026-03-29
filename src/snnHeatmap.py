import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("../figures", exist_ok=True)

jaccard_sim = np.loadtxt("jaccard_similarity.txt")
labels = np.loadtxt("snn_labels.txt", dtype=int)

unique_clusters = np.unique(labels[labels != -1])

# Clustered points sorted by cluster id; noise is appended as a separate region
cluster_order = np.argsort(labels[labels != -1])
clustered_indices = np.where(labels != -1)[0][cluster_order]
noise_indices = np.where(labels == -1)[0]
order = np.concatenate([clustered_indices, noise_indices])

reordered = jaccard_sim[np.ix_(order, order)]
sorted_labels = labels[order]

plt.figure(figsize=(10, 8))
plt.imshow(reordered, aspect='auto', cmap='BrBG', interpolation='nearest')
plt.colorbar(label='Jaccard Similarity')
plt.title("Jaccard Similarity Heatmap")
plt.xlabel("Tweet index by Cluster")
plt.ylabel("Tweet index by Cluster")

boundary = 0
for c in unique_clusters[:-1]:
    boundary += (sorted_labels == c).sum()
    plt.axhline(boundary - 0.5, color='darkgreen', linewidth=0.5)
    plt.axvline(boundary - 0.5, color='darkgreen', linewidth=0.5)

# Noise boundary line
n_clustered = (labels != -1).sum()
plt.axhline(n_clustered - 0.5, color='darkgreen', linewidth=0.5)
plt.axvline(n_clustered - 0.5, color='darkgreen', linewidth=0.5)

plt.tight_layout()
plt.savefig("../figures/snn_heatmap.png", dpi=150)
plt.show()
