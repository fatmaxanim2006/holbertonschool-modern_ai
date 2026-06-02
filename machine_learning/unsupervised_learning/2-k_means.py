#!/usr/bin/env python3
"""Module for K-means clustering"""
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Performs K-means clustering on tabular data"""
    kmeans_obj = KMeans(n_clusters=k)
    kmeans_obj.fit(X)
    return kmeans_obj.labels_, kmeans_obj.cluster_centers_
