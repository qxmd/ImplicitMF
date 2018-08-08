#!/usr/bin/env python

"""
Pre-Processing
==============
Operations to perform on a transformed matrix.
"""

from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from scipy.sparse import issparse

def _sparse_checker(X):
    if not issparse(X):
        raise TypeError("Sparse array provided is not of type csr")

def normalize_X(X, norm_type):
    """
    Normalizes the X matrix using either tfidf or bm25.
    Wrapper for Implicit's tfidf_weight and bm25_weight
    evaluation functions.

    Parameters
    ----------
    X: scipy.sparse.csr_matrix
        sparse matrix of shape (n_users, n_collections)
    norm_type: string
        can be either "bm25" or tfidf 
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized sparse csr matrix

    References
    ----------
    .. [1] bm25 and tfidf explanation:                                                    https://www.benfrederickson.com/distance-metrics/
    """
    _sparse_checker(X)
    if norm_type == "bm25":
        X = bm25_weight(X, K1=100, B=0.8)
    elif norm_type == "tfidf":
        X = tfidf_weight(X)
    else:
        raise ValueError("unknown norm_type parameter")
    return X.tocsr()