#!/usr/bin/env python3
"""Module for finding optimal K"""
from sklearn import metrics
K_Means = __import__('2-kmeans').kmeans


def optimal_k(X, max_clusters=None, random_state=0):
    """Evaluates K-Means clustering quality"""
    ks = []
    inertia_values = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        # K-Means modelini əvvəlki tapşırıqdakı funksiya ilə yaradırıq
        labels, centers = K_Means(X, k)

        # Inertia hesablanması
        # Qeyd: sklearn.cluster.KMeans obyektini tələb edir
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(X)
        
        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_scores.append(metrics.silhouette_score(X, model.labels_))

    return ks, inertia_values, silhouette_scores
