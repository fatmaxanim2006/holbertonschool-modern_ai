#!/usr/bin/env python3
"""Module for K-means clustering"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Creates and fits a K-Means clustering model"""
    model = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model
