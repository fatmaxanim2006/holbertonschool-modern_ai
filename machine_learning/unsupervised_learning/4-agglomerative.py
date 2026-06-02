#!/usr/bin/env python3
"""Module for Agglomerative Hierarchical Clustering"""
from sklearn import cluster
from sklearn import metrics


def Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering.
    """
    Apply_PCA = __import__('1-pca').Apply_PCA
    
    # 1. Dimensiya azaldılması (PCA)
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components=n_components, random_state=random_state)
    else:
        X_used = X
        
    # 2. Klasterləşdirmə (Ward linkage)
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    model.fit(X_used)
    
    # 3. Qiymətləndirmə (Silhouette score)
    score = None
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
        
    return model, X_used, score
