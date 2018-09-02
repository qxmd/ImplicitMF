#!/usr/bin/env python

"""
Unit tests for Transformer()
===========================
"""

import pytest
import numpy as np

from implicitmf.transform import Transformer
from _mock_data import create_user_item_dict

def test_transformer_input_dict_error():
    """
    Check that Transformer raises a TypeError
    if full_matrix is not a dictionary.
    """
    msg = "`user_item_dict` must be a dict"
    with pytest.raises(TypeError, match=msg):
        Transformer(user_item_dict="dict")

def test_transformer_input_bool_error():
    """
    Check that Transformer raises a TypeError
    if full_matrix is not a boolean.
    """
    ui_dict = create_user_item_dict()
    msg = "`full_matrix` must be a boolean"
    with pytest.raises(TypeError, match=msg):
        Transformer(user_item_dict=ui_dict, full_matrix="bool")

def test_transformer_attribute_type():
    """
    Check that Transformer mapper attrs are
    the correct type.
    """
    ui_dict = create_user_item_dict()
    t = Transformer(ui_dict)
    assert isinstance(t.item_inv_mapper, dict)
    assert isinstance(t.user_mapper, dict)

def test_transformer_shape():
    """
    Check that Transformer.to_sparse_array
    returns to correct shape.
    """
    ui_dict = create_user_item_dict()
    t = Transformer(ui_dict)
    X = t.to_sparse_array()
    assert X.shape == (len(ui_dict['user_id']), len(ui_dict['item_id']))

def test_transformer_num_nonzero():
    """
    Check that Transformer.to_sparse_array's
    non-zero elements are the correct length.
    """
    ui_dict = create_user_item_dict()
    t = Transformer(ui_dict)
    X = t.to_sparse_array()
    assert X.getnnz() == len(ui_dict['user_item_score'])
