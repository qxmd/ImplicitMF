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
    """
    Check that Transformer.to_sparse_array
    returns to correct shape.
    """
    uc_dict, _ = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert X.shape == (len(uc_dict['user_id']), len(uc_dict['item_id']))

def test_transformer_num_nonzero():
    """
    Check that Transformer.to_sparse_array's
    non-zero elements are the correct length.
    """
    uc_dict, _ = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert X.getnnz() == len(uc_dict['item_user_score'])

def test_transformer_loc_nonzero():
    """
    Check that Transofrmer.to_sparse_array's
    non-zero elements are in the correct location.
    """
    uc_dict, correct = gen_fetched_data()
    transform = Transformer(uc_dict)
    X = transform.to_sparse_array()
    assert np.array_equal(X.toarray(), correct)