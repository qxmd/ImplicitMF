#!/usr/bin/env python

"""
Unit tests for validation module
================================
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from implicitmf.validation import hold_out_entries, cross_val_folds
from implicit.als import AlternatingLeastSquares
from _mock_data import sparse_array

def test_hold_out_entries_type():
    """
    Check that TypeError is raised
    when sparse_arr is not sparse array.
    """
    with pytest.raises(TypeError):
        hold_out_entries(np.array([1,2,3]))

def test_hold_out_entires_shape():
    """
    Check that output array is same shape 
    as input array.
    """
    X = csr_matrix(np.array([[1,0,3],[0,1,0]]))
    X_train = hold_out_entries(X, hold_out_size=0.2)
    assert(X.shape == X_train.shape)

def test_cross_val_folds_input_error():
    """
    Check that cross_val_folds() raises a TypeError
    if n_folds is less than 2.
    """
    msg = '`n_folds` must be an integer equal to or greater than 2'
    with pytest.raises(TypeError, match=msg):
        cross_val_folds(sparse_array(), 1)

def test_cross_Val_folds_matrix_input_error():
    """
    Check that cross_val_folds() raises a TypeError
    if X is not a scipy.sparse.matrix.
    """
    msg = '`X` must be a scipy.sparse.csr_matrix'
    with pytest.raises(TypeError, match=msg):
        cross_val_folds([1,2,3], 3)

def test_cross_val_folds_size():
    """
    Check that cross_val_folds() returns
    dict of same length as num_folds.
    """
    X = sparse_array()
    n = 3
    assert(len(cross_val_folds(X, n)) == n)

def test_cross_val_folds_values():
    """
    Check that cross_val_folds() values
    are the correct format. 
    """
    X = sparse_array()
    n = 3
    output = cross_val_folds(X, n)
    assert(issparse(output[0]['train']))
    assert(issparse(output[0]['test']))