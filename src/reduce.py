# src/reduce.py
import numpy as np
import pickle
import umap
import hdbscan
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

def fit_umap_and_cluster(
    matrix_path = 'models/player_matrix.npy',
    min_cluster_size = 15,   # minimum players to form a cluster
    n_neighbors      = 15,   # UMAP: how many neighbours to consider
):
    #  1. Load player matrix 
    matrix = np.load(matrix_path)
    print(f"Player matrix loaded: {matrix.shape}")  # e.g. (800, 128)

    #  2. UMAP → 2D (for the scatter plot) ─
    print("\nFitting UMAP 2D (for visualisation)...")
    reducer_2d = umap.UMAP(
        n_components = 2,
        metric       = 'cosine',
        n_neighbors  = n_neighbors,
        min_dist     = 0.1,    # how tightly to pack points — 0.1 gives clear clusters
        random_state = 42,
    )
    coords_2d = reducer_2d.fit_transform(matrix)
    print(f"2D coords shape: {coords_2d.shape}")

    #  3. UMAP to  32D (for clustering) ─
    print("\nFitting UMAP 32D (for clustering)...")
    reducer_32d = umap.UMAP(
        n_components = 32,
        metric       = 'cosine',
        n_neighbors  = n_neighbors,
        min_dist     = 0.0,    # 0.0 for clustering — pack tightly in high-dim
        random_state = 42,
    )
    emb_32 = reducer_32d.fit_transform(matrix)
    print(f"32D embedding shape: {emb_32.shape}")

    #  4. HDBSCAN clustering on the 32D embedding ─
    print("\nClustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = 5,     # how conservative to be — lower = more clusters
        metric           = 'euclidean',
        cluster_selection_method = 'eom',  # 'eom' finds more varied cluster sizes
    )
    labels = clusterer.fit_predict(emb_32)

    #  5. Report 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"\nClusters found:  {n_clusters}")
    print(f"Noise points:    {n_noise}  ({n_noise/len(labels)*100:.1f}%)")
    print(f"Cluster sizes:")
    for cid in sorted(set(labels)):
        label = "noise" if cid == -1 else f"cluster {cid}"
        print(f"  {label:<12} {(labels==cid).sum()} players")

    #  6. Silhouette score (excluding noise) 
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        score = silhouette_score(emb_32[mask], labels[mask], metric='euclidean')
        print(f"\nSilhouette score: {score:.3f}  (> 0.3 is good, > 0.5 is excellent)")

    #  7. Save everything 
    np.save('models/coords_2d.npy', coords_2d)
    np.save('models/emb_32d.npy',   emb_32)
    np.save('models/labels.npy',    labels)
    with open('models/umap_32d.pkl',  'wb') as f: pickle.dump(reducer_32d, f)
    with open('models/hdbscan.pkl',   'wb') as f: pickle.dump(clusterer,   f)
    print("\nAll models saved to models/")

    return coords_2d, labels


if __name__ == '__main__':
    fit_umap_and_cluster(min_cluster_size=40)