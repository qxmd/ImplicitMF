#!/usr/bin/env python

"""
Unit tests for Transformer()
===========================
"""

import pytest
import numpy as np

from implicitmf.transform import Transformer
from _mock_data import gen_fetched_data, gen_bad_user_data, gen_bad_coll_data


def test_transformer_shape():
    """Test Transformer shape"""
    uc_dict, _ = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert X.shape == (len(uc_dict['user_id']), len(uc_dict['item_id']))

def test_transformer_num_nonzero():
    """Test number non-zero elements Transformer"""
    uc_dict, _ = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert X.getnnz() == len(uc_dict['item_user_score'])

def test_transformer_loc_nonzero():
    """Test location of non-zero elements of Transformer"""
    uc_dict, correct = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert np.array_equal(X.toarray(), correct)