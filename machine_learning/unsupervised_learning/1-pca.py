#!/usr/bin/env python3
"""Module for PCA application"""
from sklearn import decomposition

def Apply_PCA(X, n_components, random_state):
    """Performs PCA on tabular data using Scikit-learn"""
    pca = decomposition.PCA(n_components=n_components, random_state=random_state)
    transformed_data = pca.fit_transform(X)
    return transformed_data, pca
