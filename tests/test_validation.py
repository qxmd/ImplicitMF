#!/usr/bin/env python

"""
Unit tests for validation module
================================
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from implicitmf.validation import hold_out_entries, cross_val_folds, gridsearchCV
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
    output = cross_val_folds(sparse_array(), 3)
    assert(issparse(output[0]['train']))
    assert(issparse(output[0]['test']))

def test_gridsearchCV_output():
    """
    Check that output of gridsearchCV()
    is a pandas dataframe.
    """
    als = AlternatingLeastSquares()
    hyperparams = {
        'regularization': [0.2,0.3]
    }
    output = gridsearchCV(base_model=als, X=sparse_array(), n_folds=2, hyperparams=hyperparams)
    assert isinstance(output, pd.DataFrame)