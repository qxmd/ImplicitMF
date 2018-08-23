#!/usr/bin/env python

"""
Evaluation
==========
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy import stats
from multiprocessing.dummy import Pool as ThreadPool 
from implicit.recommender_base import RecommenderBase
from lightfm import LightFM
from implicitmf._utils import _sparse_checker

def get_X_rec(model, users, X, k, user_features=None, num_threads=1):
    """
    Populates the user-item utility matrix with user recommendations.
    
    Parameters
    ----------
    model : implicit.recommender_base.RecommenderBase or lightfm.lightfm.LightFM
        trained model that we want recommendations from 
    users : iterable
        collection of users to make recommendations for
    X : scipy.sparse.csr_matrix
        utility matrix of shape (u, i) where u is the number of users
        and i is the number of items
    num_threads: int
        the number of threads to use in computation of recommendations 
        (only works with LightFM)
        
    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix where each user has top k items stored with associated scores

    Note
    ----
    This function is for development purposes.
    """
    user_ind = list()
    item_ind = list()
    data = list()
   
    if isinstance(model, RecommenderBase):
        def gen_rec(user):
            rec_scores = model.recommend(user, X, N=k)
            ind = [i[0] for i in rec_scores]
            scores = [i[1] for i in rec_scores]
            return np.array([user + np.zeros(k), ind, scores]).T

    elif isinstance(model, LightFM):
        items = np.arange(X.shape[1], dtype=np.int32)
        def gen_rec(user):
            scores = model.predict(user_ids=user, user_features=user_features, item_ids=items)
            ind = scores.argsort()[-k:]
            return np.array([user + np.zeros(k), ind, scores[ind]]).T
         
    else: 
        raise TypeError('Model must be instance of `LightFM` or `RecommenderBase`')
    
    pool = ThreadPool(num_threads)
    results = pool.map(gen_rec, users)
    pool.close()
    pool.join()
    
    user_item_score = np.vstack(results)
    user_ind = user_item_score[:,0]
    item_ind = user_item_score[:,1]
    data = user_item_score[:,2]
    recommendations = csr_matrix((data, (user_ind, item_ind)), shape=X.shape)
        
    return recommendations

def precision_at_k(X_test, X_rec, k):
    """
    Computes mean precision@k

    Parameters
    ----------
    X_test : scipy.sparse.csr_matrix
    X_rec : scipy.sparse.csr_matrix
        user-collection matrix with each row representing k recommendations for a user
    k : int
        number of recommendations to provide for each user
    
    Returns
    -------
    float
        precision@k
    """
    _sparse_checker(X_test, '`X_test`')
    _sparse_checker(X_rec, '`X_rec`')

    if X_test.shape != X_rec.shape:
        raise TypeError('`X_test` must be the same shape as `X_rec`')
    
    overlap = X_test.multiply(X_rec)
    overlap.eliminate_zeros()
    user_nnz = overlap.getnnz(axis=1)
    mean_prec = np.mean(user_nnz / k)
    print('Mean precision@{0:d} of {1:.6f}'.format(k, mean_prec))
    return np.array(user_nnz / k).reshape(-1)
