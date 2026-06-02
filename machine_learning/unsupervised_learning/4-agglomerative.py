#!/usr/bin/env python3
"""Module for Agglomerative Hierarchical Clustering"""
from sklearn import cluster
from sklearn import metrics


def Agglomerative_Clustering(X, n_clusters, random_state=0, n_components=0):
    """
    Performs Agglomerative hierarchical clustering.
    """
    # Dinamik import
    Apply_PCA = __import__('1-pca').Apply_PCA
    
    # PCA tətbiqi yalnız n_components > 0 olduqda baş verməlidir
    if n_components > 0:
        X_used, _ = Apply_PCA(X, n_components=n_components, random_state=random_state)
    else:
        # PCA tətbiq olunmadıqda orijinal X istifadə olunur
        X_used = X
        
    # Klasterləşdirmə
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    model.fit(X_used)
    
    # Silhouette score hesablama (n_clusters > 1 şərti ilə)
    score = None
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
        
    return model, X_used, score
