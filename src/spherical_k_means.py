import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from collections import Counter


#load feature matrix
X = load_npz("tweet_features.npz")
feature_names = np.loadtxt("feature_names.txt", dtype=str, delimiter="\t")

#normalize to do spherical
X_normalized = normalize(X, norm='l2', axis=1)
X_reduced = TruncatedSVD(n_components=100, random_state=42).fit_transform(X_normalized)

#cosine similarity matrix
cos_sim = cosine_similarity(X_reduced)
cos_dist = 1 - cos_sim  # convert similarity to distance for SSE/BSS

max_k = 10
sil_scores = []
SSE = []

for k in range(2, max_k+1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_normalized)
    
    # silhouette (cosine metric works on sparse)
    score = silhouette_score(X_normalized, labels, metric='cosine')
    sil_scores.append(score)
    
    # SSE for sparse matrix
    sse = 0
    for cluster_id in range(k):
        cluster_points = X_normalized[labels == cluster_id]
        center = kmeans.cluster_centers_[cluster_id]
        sse += np.sum((cluster_points.toarray() - center)**2)
    SSE.append(sse)

#sillouhette and sse based on cluster size
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(2, max_k+1), sil_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")

plt.subplot(1,2,2)
plt.plot(range(2, max_k+1), SSE, marker='o', color='orange')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.title("SSE vs k")

plt.tight_layout()
plt.savefig("kmeans_cluster_quality_metrics.png")
plt.show()

#pick the k with highest sillhouette score
best_k = np.argmax(sil_scores) + 2
print(f"Optimal number of clusters based on silhouette score: {best_k}")

#run k means spherical
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_final = kmeans_final.fit_predict(X_normalized)
np.savetxt("tweet_clusters.txt", labels_final, fmt='%d')

#distribution of cluster sizes
unique, counts = np.unique(labels_final, return_counts=True)
print("Cluster sizes:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} tweets")

plt.figure(figsize=(8,5))
plt.bar(unique, counts, color='skyblue')
plt.xlabel("Cluster")
plt.ylabel("Number of Tweets")
plt.title("Cluster Size Distribution")
plt.xticks(unique)
plt.tight_layout()
plt.savefig("kmeans_cluster_size_distribution.png")
plt.show()

# reorder for heatmap
sorted_idx = np.argsort(labels_final)
sorted_dist = cos_dist[sorted_idx][:, sorted_idx]

plt.figure(figsize=(10,10))
sns.heatmap(sorted_dist, cmap="viridis")
plt.title("Cosine Distance Heatmap by Cluster Membership")
plt.xlabel("Tweets (sorted by cluster)")
plt.ylabel("Tweets (sorted by cluster)")
plt.tight_layout()
plt.savefig("kmeans_cluster_heatmap.png")
plt.show()

#cluster selection
unique, counts = np.unique(labels_final, return_counts=True)
largest_cluster = unique[np.argmax(counts)]
print(f"Largest cluster ID: {largest_cluster}, size: {np.max(counts)}")

#tweet indices
cluster_idx = np.where(labels_final == largest_cluster)[0]

#sum all the feature counts
cluster_matrix = X[cluster_idx]   # sparse matrix of cluster tweets
word_sums = np.array(cluster_matrix.sum(axis=0)).flatten()
word_freq = pd.DataFrame({'word': feature_names, 'score': word_sums})

#get top 10 words
top_words = word_freq.sort_values('score', ascending=False).head(10)
print("Top 10 words in cluster:")
print(top_words)

labels_pred = labels_final
num_clusters = len(np.unique(labels_pred))
N = len(labels_pred)

entropy_total = 0
purity_total = 0

# Use top word in each tweet as proxy "true label"
# Convert sparse matrix to dense for word lookup
X_dense = X.toarray()
top_words_per_tweet = [feature_names[np.argmax(row)] for row in X_dense]

for k in range(num_clusters):
    idx = np.where(labels_pred == k)[0]
    cluster_size = len(idx)
    
    # Get the top words for tweets in this cluster
    words_in_cluster = [top_words_per_tweet[i] for i in idx]
    
    counts = np.array(list(Counter(words_in_cluster).values()))
    probs = counts / counts.sum()

    # entropy for this cluster
    H = -np.sum(probs * np.log2(probs))
    entropy_total += (cluster_size / N) * H

    # purity for this cluster
    purity_total += (cluster_size / N) * probs.max()

print(f"Total Entropy (proxy): {entropy_total:.4f}")
print(f"Total Purity (proxy): {purity_total:.4f}")