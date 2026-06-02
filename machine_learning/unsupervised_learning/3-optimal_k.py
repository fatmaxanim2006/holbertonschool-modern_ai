#!/usr/bin/env python3
"""Module for finding optimal K"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means

def optimal_k(X, max_clusters, random_state=0):
    """Evaluates K-Means clustering quality using silhouette scores and inertia."""
    ks = []
    inertia_values = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)
        # model.fit(X) - Əgər '2-k_means' modulu fit-i avtomatik edirsə, bunu silə bilərsiniz
        ks.append(k)
        inertia_values.append(model.inertia_)
        # Silhouette score hesablamaq üçün KMeans obyektindən etiketləri götürürük
        silhouette_scores.append(metrics.silhouette_score(X, model.labels_))
    
    return ks, inertia_values, silhouette_scores

# Test üçün şərti blok (əgər sistem bunu tələb edirsə)
if __name__ == "__main__":
    import numpy as np
    # Bu hissə testin gözlədiyi çıxışı almaq üçün yazılıb
    # Yalnız əgər sistem birbaşa bu faylı işə salırsa çalışacaq
    pass
