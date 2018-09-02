#!/usr/bin/env python

"""
Unit tests for datasets module
==============================
"""

import pytest
import pandas as pd
import numpy as np

from implicitmf.datasets import movielens

def test_movielens_output():
    """
    Check that movielens() returns
    the correct output type.
    """
    data = movielens(type='df')
    assert(isinstance(data, pd.DataFrame))
    data = movielens(type='array')
    assert(isinstance(data, np.array))
    
def test_movielens_columns():
    """
    Check that movielens() returns a dataframe
    with the correct column values.
    """
    movielens_columns = ['user_id', 'item_id', 'ratings']
    data = movielens(type='df')
    assert(data.columns.values == movielens_columns)
