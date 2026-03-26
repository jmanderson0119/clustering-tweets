import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("../figures", exist_ok=True)

cos_sim = np.loadtxt("cosine_similarity.txt")
labels = np.loadtxt("avg_link_labels.txt", dtype=int)

unique_clusters = np.unique(labels)

# Reorders by cluster
order = np.argsort(labels)
reordered = cos_sim[np.ix_(order, order)]
sorted_labels = labels[order]

plt.figure(figsize=(10, 8))
plt.imshow(reordered, aspect='auto', cmap='BrBG', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.title("Cosine Similarity Heatmap")
plt.xlabel("Tweet index by Cluster")
plt.ylabel("Tweet index by Cluster")

# Cluster boundaries
boundary = 0
for c in unique_clusters[:-1]:
    boundary += (sorted_labels == c).sum()
    plt.axhline(boundary - 0.5, color='darkgreen', linewidth=0.5)
    plt.axvline(boundary - 0.5, color='darkgreen', linewidth=0.5)

plt.tight_layout()
plt.savefig("../figures/avg_link_heatmap.png", dpi=150)
plt.show()
