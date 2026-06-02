#!/usr/bin/env python3
"""Module for data standardization"""
from sklearn import preprocessing

def Standardize(x):
    """Standardizes input data using StandardScaler"""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x)
