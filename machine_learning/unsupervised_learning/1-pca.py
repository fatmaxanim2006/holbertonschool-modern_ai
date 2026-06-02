#!/usr/bin/env python3
"""Module for PCA dimensionality reduction"""
from sklearn import decomposition


def Apply_PCA(X, n_components=None, random_state=None):
    """Performs PCA on tabular data using Scikit-learn"""
    pca = decomposition.PCA(n_components=n_components,
                            random_state=random_state)
    return pca.fit_transform(X), pca
