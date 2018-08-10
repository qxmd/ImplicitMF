#!/usr/bin/env python

"""
Mock data for unit tests
========================
- `sparse_matrix`: a scipy sparse matrix in csr ("compressed sparse row") format
"""

from scipy.sparse import rand
from implicitmf.transform import Transformer
import numpy as np

# ---------------------------------------------------------------------
# Transformer() input
# ---------------------------------------------------------------------

def gen_fetched_data():
    """
    Function to generate inputs of Transformer
    for basic unit testing
    """
    distinct_users = np.array([111, 222, 333, 444, 555, 666, 777, 888])
    distinct_colls = np.array([201, 202, 203, 304, 305, 306])
    fetched_cs = np.array([(111, 201, 1), (333, 203, 1), (777, 306, 1)])
    correct_dense = np.zeros((len(distinct_users), len(distinct_colls)))
    correct_dense[0, 0] = 1
    correct_dense[2, 2] = 1
    correct_dense[6, 5] = 1
    uc_dict = {
        'item_user_score': fetched_cs,
        'user_id': distinct_users,
        'item_id':distinct_colls
        }
    return uc_dict, correct_dense

def gen_bad_user_data():
    """Helper to generate bad user id data"""
    bad_user_dict = {
        'item_user_score': np.array([(12, 12, 1), (13, 14, 0)]),
        'user_id': np.array([1, 12]),
        'item_id': np.array([12, 14])
    }
    return bad_user_dict

def gen_bad_coll_data():
    """Helper to generate bad item id data"""
    bad_user_dict = {
        'item_user_score': np.array([(12, 12, 1), (13, 14, 0)]),
        'user_id': np.array([12, 13]),
        'item_id': np.array([1, 14])
    }
    return bad_user_dict


# ---------------------------------------------------------------------
# Sparse matrix
# ---------------------------------------------------------------------

def sparse_array():
    """
    Uses gen_fetched_data to transform into
    a sparse array.
    """
    uc_dict, _ = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    return X

# ---------------------------------------------------------------------
# Recommendations dictionary
# ---------------------------------------------------------------------

def recommendations_dict():
    """
    Creates a rec_dict with 100 users and 20
    recommendations per user. Also generates a
    user_sub_dict which samples from rec_dict users
    and generates a new dictionary with 'already
    subscribed' items.
    """
    rec_dict = dict()
    user_ids = np.random.randint(1000,size=100)
    rec_ids = np.random.randint(1000, size=(100,20))
    scores = np.random.rand(100,20)
    for i in user_ids:
        rec_dict[i] = dict()
        rec_dict[i]['recs'] = np.random.randint(1000, size=20)
        rec_dict[i]['score'] = np.random.rand(20) 
    # sample from rec_dict users and generate new dictionary
    # with their "already subscribed" items
    user_sub_keys = np.random.choice(list(rec_dict.keys()), replace=False, size=10)
    user_sub_dict = {k: rec_dict[k] for k in user_sub_keys}
    for k, v in user_sub_dict.items():
        user_sub_dict[k] = np.random.choice(v['recs'], replace=False,size=3)
    return rec_dict, user_sub_dict