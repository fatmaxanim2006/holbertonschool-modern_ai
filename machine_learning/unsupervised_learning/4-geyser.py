#!/usr/bin/env python3
"""Module for performing K-Means clustering on geyser data"""
import numpy as np
K_Means = __import__('2-k_means').K_Means

def geyser():
    """Performs K-Means clustering on the geyser dataset"""
    data = np.load('geyser.npy')
    
    # 3 klaster üçün modeli işə salırıq
    model = K_Means(data, n_clusters=3)
    
    # Düzgün atribut adları (scikit-learn standartı olan '_' ilə bitənlər)
    return model.labels_, model.cluster_centers_
