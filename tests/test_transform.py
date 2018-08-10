#!/usr/bin/env python

"""
Unit tests for Transformer()
===========================
"""

import pytest
import numpy as np

from implicitmf.transform import Transformer
from _mock_data import gen_fetched_data, gen_bad_user_data, gen_bad_coll_data

def test_transformer_input_dict_error():
    """
    Check that Transformer raises a TypeError
    if full_matrix is not a dictionary.
    """
    msg = "user_item_dict parameter must be a dict."
    with pytest.raises(TypeError, match=msg):
        Transformer(user_item_dict="dict")

def test_transformer_input_bool_error():
    """
    Check that Transformer raises a TypeError
    if full_matrix is not a boolean.
    """
    uc_dict, _ = gen_fetched_data()
    msg = "full_matrix parameter must be a boolean."
    with pytest.raises(TypeError, match=msg):
        Transformer(user_item_dict=uc_dict, full_matrix="bool")

def test_transformer_attribute_type():
    """
    Check that Transformer mapper attrs are
    the correct type.
    """
    uc_dict, _ = gen_fetched_data()
    t = Transformer(uc_dict)
    assert isinstance(t.item_inv_mapper, dict)
    assert isinstance(t.user_mapper, dict)

def test_transformer_shape():
    """
    Check that Transformer.to_sparse_array
    returns to correct shape.
    """
    uc_dict, _ = gen_fetched_data()
    t = Transformer(uc_dict)
    X = t.to_sparse_array()
    assert X.shape == (len(uc_dict['user_id']), len(uc_dict['item_id']))

def test_transformer_num_nonzero():
    """
    Check that Transformer.to_sparse_array's
    non-zero elements are the correct length.
    """
    uc_dict, _ = gen_fetched_data()
    t = Transformer(uc_dict)
    X = t.to_sparse_array()
    assert X.getnnz() == len(uc_dict['item_user_score'])

def test_transformer_loc_nonzero():
    """
    Check that Transofrmer.to_sparse_array's
    non-zero elements are in the correct location.
    """
    uc_dict, correct = gen_fetched_data()
    t = Transformer(uc_dict)
    X = t.to_sparse_array()
    assert np.array_equal(X.toarray(), correct)