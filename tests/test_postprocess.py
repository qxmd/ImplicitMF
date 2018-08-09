#!/usr/bin/env python

"""
Unit tests for post-processing module
=====================================
"""

import pytest
import numpy as np

from implicitmf.postprocess import remove_subscribed_items

def test_remove_subscribed_items_output():
    """
    Check that remove_subscribed_items() returns
    the correct output type.
    """
    pass

def test_remove_subscribed_items_dict_error():
    """
    Check that remove_subscribed_items() raises a
    TypeError if rec_dict/user_sub_dict is not a dict.
    """
    pass