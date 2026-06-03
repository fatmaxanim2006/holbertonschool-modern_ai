#!/usr/bin/env python3
"""Module for performing K-Means clustering on geyser data"""
import numpy as np
K_Means = __import__('2-k_means').K_Means

def geyser():
    """Performs K-Means clustering on the geyser dataset"""
    # Verilənləri yükləyirik
    data = np.load('geyser.npy')
    
    # K-Means modelini qururuq (tələb olunan k=3 klaster üçün)
    model = K_Means(data, n_clusters=3)
    
    # Klasterləşdirməni yerinə yetiririk
    labels = model.labels_
    
    return labels, model
