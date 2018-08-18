#!/usr/bin/env python

"""
Pre-Processing
==============
Operations to perform on a transformed matrix.
"""

from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from implicitmf._utils import _sparse_checker

def normalize_X(X, norm_type):
    """
    Normalizes the X matrix using either tfidf or bm25.
    Wrapper for tfidf_weight and bm25_weight functions from
    the :mod:`implicit:implicit.nearest_neighbours` module.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        sparse matrix of shape (n_users, n_collections)
    norm_type : str
        can be either "bm25" or tfidf 
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized sparse csr matrix

    References
    ----------
    .. [1] bm25 and tfidf explanation: https://www.benfrederickson.com/distance-metrics/
    .. [2] https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx
    """
    _sparse_checker(X)
    if norm_type == "bm25":
        X = bm25_weight(X, K1=100, B=0.8)
    elif norm_type == "tfidf":
        X = tfidf_weight(X)
    else:
        raise ValueError("Unknown `norm_type` parameter.")
    return X.tocsr()