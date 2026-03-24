import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#feature matrix
X = load_npz("tweet_features.npz")

#cosine similarity
cos_sim = cosine_similarity(X)
np.savetxt("cosine_similarity.txt", cos_sim)

#jaccard similarity, binarize the matrix so it doesn't take a year to run
X_bin = (X > 0).astype(int)  # binarize
intersection = X_bin @ X_bin.T
row_sums = X_bin.sum(axis=1).A1
union = row_sums[:, None] + row_sums[None, :] - intersection.A
jaccard_sim = intersection.A / union
np.savetxt("jaccard_similarity.txt", jaccard_sim)

#histograms with a log scale
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(cos_sim.flatten(), bins=50, color='skyblue', edgecolor='black', log=True)
plt.title("Cosine Similarity Distribution (log scale)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency (log scale)")

plt.subplot(1,2,2)
plt.hist(jaccard_sim.flatten(), bins=50, color='lightgreen', edgecolor='black', log=True)
plt.title("Jaccard Similarity Distribution (log scale)")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Frequency (log scale)")

plt.tight_layout()
plt.show()
