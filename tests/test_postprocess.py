#!/usr/bin/env python

"""
Unit tests for post-processing module
=====================================
"""

import pytest
import numpy as np

from implicitmf.postprocess import remove_subscribed_items
from _mock_data import recommendations_dict

def test_remove_subscribed_items_output():
    """
    Check that remove_subscribed_items() returns
    the correct output type.
    """
    rec_dict, user_sub_dict = recommendations_dict()
    output = remove_subscribed_items(rec_dict, user_sub_dict)
    assert isinstance(output, dict)

def test_remove_subscribed_items_user_sub_dict_error():
    """
    Check that remove_subscribed_items() raises a
    TypeError if rec_dict/user_sub_dict is not a dict.
    """
    rec_dict, user_sub_dict = recommendations_dict() 
    msg = "`user_sub_dict` must be a dict"
    with pytest.raises(TypeError, match=msg):
        remove_subscribed_items(rec_dict, "testing 123")

def test_remove_subscribed_items_rec_dict_error():
    """
    Check that remove_subscribed_items() raises a
    TypeError if rec_dict/user_sub_dict is not a dict.
    """
    rec_dict, user_sub_dict = recommendations_dict() 
    msg = "`rec_dict` must be a dict"
    with pytest.raises(TypeError, match=msg):
        remove_subscribed_items([1,2,3,4], user_sub_dict)


def test_remove_subscribed_items_list_error():
    """
    Check that remove_subscribed_items() raises a
    TypeError if unwanted_items is not a list.
    """
    rec_dict, user_sub_dict = recommendations_dict() 
    msg = "`unwanted_items` must be a list"
    with pytest.raises(TypeError, match=msg):
        remove_subscribed_items(rec_dict, user_sub_dict, unwanted_items=42)