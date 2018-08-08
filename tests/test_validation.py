#!/usr/bin/env python

"""
Unit tests for validation module
================================
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix

from implicitmf.validation import hold_out_entries, cross_val_folds

def test_hold_out_entries_type():
    """Check that TypeError thrown when sparse_arr is not sparse array"""
    with pytest.raises(TypeError):
        hold_out_entries(np.array([1,2,3]))

def test_hold_out_entires_shape():
    """Check that output array is same shape as input array"""
    X = csr_matrix(np.array([[1,0,3],[0,1,0]]))
    X_train = hold_out_entries(X, hold_out_size=0.2)
    assert(X.shape == X_train.shape)