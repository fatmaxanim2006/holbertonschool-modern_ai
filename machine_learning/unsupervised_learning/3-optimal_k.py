#!/usr/bin/env python3
"""Module for finding optimal K"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters=None, random_state=0):
    """
    Evaluates K-Means clustering quality using silhouette scores and inertia.

    Args:
        X (numpy.ndarray): Tabular data.
        max_clusters (int): Maximum number of clusters.
        random_state (int): Random seed.

    Returns:
        tuple: (ks, inertia_values, silhouette_scores)
    """
    ks = []
    inertia_values = []
    silhouette_scores = []

    # max_clusters-in None olma ehtimalına qarşı yoxlanış
    if max_clusters is None:
        return ks, inertia_values, silhouette_scores

    for k in range(2, max_clusters + 1):
        model = K_Means(n_clusters=k, random_state=random_state)
        model.fit(X)
        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_scores.append(metrics.silhouette_score(X, model.labels_))

    return ks, inertia_values, silhouette_scores
