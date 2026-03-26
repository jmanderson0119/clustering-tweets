import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.metrics import silhouette_samples


#feature matrix and binarized version for jaccard
X = load_npz("tweet_features.npz")
X_bin = (X > 0).astype(int)

#do the jaccard similarity manually
intersection = X_bin @ X_bin.T
row_sums = X_bin.sum(axis=1).A1
union = row_sums[:, None] + row_sums[None, :] - intersection.A
jaccard_sim = intersection.A / union
jaccard_dist = 1 - jaccard_sim  # distance matrix

dists = jaccard_dist[np.triu_indices_from(jaccard_dist, k=1)]
#we chose eps by looking at a graph of the sorted distances and picking a value that seemed 
#to separate the lower distances from the higher ones
eps = .77  

#dbscan
min_samples = 5
db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
labels = db.fit_predict(jaccard_dist)

#summary of clusters
unique, counts = np.unique(labels, return_counts=True)
noise_count = (labels == -1).sum()
print(f"Noise: {noise_count} tweets")
for u, c in zip(unique, counts):
    if u != -1:
        print(f"Cluster {u}: {c} tweets")

#cluster sizes vizualization
plt.figure(figsize=(8,5))
cluster_sizes = [c for u,c in zip(unique, counts) if u != -1]
cluster_ids = [u for u in unique if u != -1]
plt.bar(cluster_ids, cluster_sizes, color='skyblue')
plt.xlabel("Cluster ID")
plt.ylabel("Number of Tweets")
plt.title("DBSCAN Cluster Sizes")
plt.tight_layout()
plt.savefig("dbscan_cluster_sizes.png")

#compute sse and silhouette scores
if len(set(labels)) > 1 and len(set(labels)) - (1 if -1 in labels else 0) > 1:
    sil_samples = silhouette_samples(jaccard_dist, labels, metric='precomputed')
    sil_list = []
    for u in cluster_ids:
        idx = np.where(labels == u)[0]
        sil_list.append(sil_samples[idx].mean())
else:
    sil_list = [np.nan]*len(cluster_ids)

#compute SSE
sse_list = []
for u in cluster_ids:
    idx = np.where(labels == u)[0]
    cluster_points = X[idx].toarray()
    center = cluster_points.mean(axis=0)
    sse = np.sum((cluster_points - center) ** 2)
    sse_list.append(sse)

#visualize SSE
plt.figure(figsize=(6,5))
plt.bar(cluster_ids, sse_list, color='orange')
plt.xlabel("Cluster ID")
plt.ylabel("SSE")
plt.title("SSE by Cluster")
plt.tight_layout()
plt.savefig("dbscan_sse.png")

#visualize silhouette using the sil_samples computed earlier
plt.figure(figsize=(6,5))
plt.bar(cluster_ids, sil_list, color='green')
plt.xlabel("Cluster ID")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Cluster")
plt.tight_layout()
plt.savefig("dbscan_silhouette.png")

#heatmap
sorted_idx = np.argsort(labels)
sorted_dist = jaccard_dist[sorted_idx][:, sorted_idx]
plt.figure(figsize=(10,10))
sns.heatmap(sorted_dist, cmap="viridis")
plt.title("Jaccard Distance Heatmap (DBSCAN Clusters)")
plt.xlabel("Tweets (sorted by cluster)")
plt.ylabel("Tweets (sorted by cluster)")
plt.tight_layout()
plt.savefig("dbscan_heatmap.png")
plt.show()
#save the labels
np.savetxt("dbscan_tweet_clusters.txt", labels, fmt='%d')

sil_list = []

for u in cluster_ids:
    idx = np.where(labels == u)[0]
    if len(idx) > 1:  # silhouette only valid for >1 sample
        cluster_points = X[idx].toarray()
        # compute silhouette of points vs cluster assignment (all same label = silhouette ~ 0)
        sil = silhouette_score(cluster_points, np.zeros(len(idx)), metric='cosine')
        sil_list.append(sil)
    else:
        sil_list.append(np.nan)
#silhouette score by cluster
plt.figure(figsize=(6,5))
plt.bar(cluster_ids, sil_list, color='green')
plt.xlabel("Cluster ID")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Cluster")
plt.tight_layout()
plt.savefig("dbscan_silhouette.png")

#heatmap of jaccard distance sorted by cluster
sorted_idx = np.argsort(labels)
sorted_dist = jaccard_dist[sorted_idx][:, sorted_idx]

plt.figure(figsize=(10,10))
sns.heatmap(sorted_dist, cmap="viridis")
plt.title("Jaccard Distance Heatmap (DBSCAN Clusters)")
plt.xlabel("Tweets (sorted by cluster)")
plt.ylabel("Tweets (sorted by cluster)")
plt.tight_layout()
plt.savefig("dbscan_heatmap.png")
plt.show()

