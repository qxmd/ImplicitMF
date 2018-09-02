#!/usr/bin/env python

"""
Pre-Processing
==============
Operations to perform on a transformed matrix.
"""
import numpy as np
import pandas as pd

from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from implicitmf._utils import _sparse_checker

def dict_converter(ratings, unique_users=None, unique_items=None):
    """
    Converts a pd.DataFrame or np.ndarray into a dictionary
    that can be passed into the Transform class.

    Parameters
    ----------
    ratings : pd.DataFrame
        dataframe with 3 columns: user_id, item_id, rating
    unique_users : pd.DataFrame or np.ndarray
        one-dimensional array that represents unique users in
        the dataset
    unique_items : pd.DataFrame or np.ndarray
        one-dimensional array that represents unique items in
        the dataset

    Returns
    -------
    dict
        dictionary with 3 keys: 'user_item_score',
        'distinct_items', 'distinct_users'
    """
    if(ratings.shape[1] != 3):
        raise ValueError("ratings must have 3 columns")
    user_item_dict = dict()
    if isinstance(ratings, pd.DataFrame):
        ratings = ratings.values
    if unique_users is None:
        unique_users = np.unique(ratings[:,0])
    if unique_items is None:
        unique_users = np.unique(ratings[:,1])
    user_item_score = [tuple(i) for i in ratings]  
    user_item_dict['item_id'] = unique_items
    user_item_dict['user_id'] = unique_users
    user_item_dict['user_item_score'] = user_item_score
    return user_item_dict

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
    _sparse_checker(X, '`X`')
    if norm_type == "bm25":
        X = bm25_weight(X, K1=100, B=0.8)
    elif norm_type == "tfidf":
        X = tfidf_weight(X)
    else:
        raise ValueError("Unknown `norm_type` parameter.")
    return X.tocsr()