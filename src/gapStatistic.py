import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
os.makedirs("../figures", exist_ok=True)

cos_sim = np.loadtxt("cosine_similarity.txt")
X = load_npz("tweet_features.npz").toarray()

dist_matrix = 1.0 - cos_sim
np.clip(dist_matrix, 0, None, out=dist_matrix)
condensed = squareform(dist_matrix, checks=False)

dendro = linkage(condensed, method='average')

def compute_wcss(X, labels):
    wcss = 0.0
    for lbl in np.unique(labels):
        cluster_points = X[labels == lbl]
        centroid = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

k_range = range(2, 1000)
B = 20

# WCSS for observed data
wcss_orig = []
for k in k_range:
    labels_k = fcluster(dendro, k, criterion='maxclust')
    wcss_orig.append(compute_wcss(X, labels_k))
    if k % 100 == 0:
        print(f"observed k: {k}")
wcss_orig = np.array(wcss_orig)

# WCSS for reference datasets
wcss_refs = np.zeros((len(k_range), B))
mins = X.min(axis=0)
maxs = X.max(axis=0)

for b in range(B):
    X_ref = np.random.uniform(low=mins, high=maxs, size=X.shape)
    cos_sim_ref = cosine_similarity(X_ref)
    dist_ref = 1.0 - cos_sim_ref
    np.clip(dist_ref, 0, None, out=dist_ref)
    condensed_ref = squareform(dist_ref, checks=False)
    dendro_ref = linkage(condensed_ref, method='average')
    for i, k in enumerate(k_range):
        labels_ref = fcluster(dendro_ref, k, criterion='maxclust')
        wcss_refs[i, b] = compute_wcss(X_ref, labels_ref)
    print(f"reference dataset {b + 1}/{B}")

# gap statistic and standard error
log_wcss_ref = np.log(wcss_refs).mean(axis=1)
gap = log_wcss_ref - np.log(wcss_orig)
s_k = np.sqrt(((np.log(wcss_refs) - log_wcss_ref[:, None]) ** 2).sum(axis=1) / B) * np.sqrt(1 + 1 / B)

criterion = gap[:-1] - (gap[1:] - s_k[1:])

opt_k = None
for i in range(len(criterion)):
    if criterion[i] >= 0:
        opt_k = list(k_range)[i]
        break

# Falls back to argmax if hte gap criterion doesn't trigger within k range
if opt_k is None:
    opt_k = list(k_range)[np.argmax(gap)]

print(f"\nOptimal k: {opt_k}")
k_vals = list(k_range)[:-1]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Gap statistic
axes[0].errorbar(list(k_range), gap, yerr=s_k, fmt='-', color='black',
                 linewidth=0.6, elinewidth=0.4, capsize=1.5,
                 errorevery=10, label='gap ± s_k')
axes[0].axvline(opt_k, color='purple', linestyle='--', linewidth=1.0,
                label=f'optimal k: {opt_k}')
axes[0].set_ylabel("Gap Statistic")
axes[0].set_title("Gap Statistic for Hierarchical Clustering")
axes[0].legend()

# Criterion value
axes[1].plot(k_vals, criterion, color='black', linewidth=0.6)
axes[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
axes[1].axvline(opt_k, color='purple', linestyle='--', linewidth=1.0,
                label=f'zero crossing at k: {opt_k}')
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("gap(k) − (gap(k+1) − s(k+1))")
axes[1].set_title("Gap Criterion Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("../figures/hier_gap_stat.png", dpi=600)
plt.show()
