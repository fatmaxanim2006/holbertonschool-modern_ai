#!/usr/bin/env python3
"""Module for Agglomerative Hierarchical Clustering"""
from sklearn import cluster
from sklearn import metrics


def Agglomerative_Clustering(X, n_clusters, random_state=0, n_components=0):
    """
    Performs Agglomerative hierarchical clustering.
    """
    # Dinamik import tələbi
    Apply_PCA = __import__('1-pca').Apply_PCA
    
    # 1. Dimensiya azaldılması
    if n_components > 0:
        X_used, _ = Apply_PCA(X, n_components=n_components, random_state=random_state)
    else:
        X_used = X
        
    # 2. Klasterləşdirmə (ward linkage standartdır)
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    model.fit(X_used)
    
    # 3. Silhouette score hesablama
    # n_clusters=1 olduqda silhouette_score işləmir, ona görə None qaytarırıq
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None
        
    return model, X_used, score
