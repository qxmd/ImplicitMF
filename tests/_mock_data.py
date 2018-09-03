#!/usr/bin/env python

"""
Mock data for unit tests
========================
- `sparse_matrix`: a scipy sparse matrix in csr ("compressed sparse row") format
"""

from scipy.sparse import rand
from implicitmf.transform import Transformer
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Dataframe
# ---------------------------------------------------------------------

def create_ratings_df():
    """
    Creates dataframe with columns:
    'user_id', 'item_id', 'ratings'
    """
    size = 10
    d = dict()
    d['user_id'] = np.random.choice(1000, size)
    d['item_id'] = np.random.choice(1000, size)
    d['ratings'] = np.random.choice(5, size)
    data = pd.DataFrame(d)
    return data

# ---------------------------------------------------------------------
# Transformer() input
# ---------------------------------------------------------------------

def create_user_item_dict():
    """
    Generates inputs of Transformer
    for basic unit testing.
    """
    distinct_users = np.array([111, 222, 333, 444, 555, 666, 777, 888])
    distinct_items = np.array([201, 202, 203, 304, 305, 306])
    user_item_score = np.array([(111, 201, 1), (333, 203, 1), (777, 306, 1)])
    user_item_dict = dict()
    user_item_dict['item_id'] = distinct_users
    user_item_dict['user_id'] = distinct_items
    user_item_dict['user_item_score'] = user_item_score
    return user_item_dict


# ---------------------------------------------------------------------
# Sparse matrix
# ---------------------------------------------------------------------

def sparse_array():
    """
    Uses create_user_item_dict() to transform into
    a sparse array.
    """
    ui_dict = create_user_item_dict()
    transform = Transformer(ui_dict)
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
    user_ids = np.random.randint(1000, size=100)
    rec_ids = np.random.randint(1000, size=(100, 20))
    scores = np.random.rand(100, 20)
    for i in user_ids:
        rec_dict[i] = dict()
        rec_dict[i]['recs'] = np.random.randint(1000, size=20)
        rec_dict[i]['score'] = np.random.rand(20)
    # sample from rec_dict users and generate new dictionary
    # with their "already subscribed" items
    user_sub_keys = np.random.choice(
        list(rec_dict.keys()), replace=False, size=10)
    user_sub_dict = {k: rec_dict[k] for k in user_sub_keys}
    for k, v in user_sub_dict.items():
        user_sub_dict[k] = np.random.choice(v['recs'], replace=False, size=3)
    return rec_dict, user_sub_dict
