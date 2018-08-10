#!/usr/bin/env python

"""
Unit tests for pre-processing module
====================================
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from implicitmf.preprocess import normalize_X
from _mock_data import sparse_array


def test_normalize_X_output():
    """
    Check that output of normalize_X()
    is a scipy.sparse.csr matrix.
    """
    X = sparse_array()
    output = normalize_X(X, norm_type="bm25")
    assert isinstance(output, csr_matrix)
    assert output.shape == X.shape

def test_normalize_X_incorrect_sparse_matrix():
    """
    Check that normalize_X() raises a
    TypeError if X is not the correct format.
    """
    msg = "X must be a sparse matrix of type csr."
    with pytest.raises(TypeError, match=msg):
        normalize_X(X="hello", norm_type="bm25")

def test_normalize_X_incorrect_norm_type():
    """
    Check that normalize_X() raises a ValueError
    if norm_type is not one of bm25 or tfidf.
    """
    msg = "Unknown norm_type parameter."
    with pytest.raises(ValueError, match=msg):
        normalize_X(X=sparse_array(), norm_type="bm2000")

