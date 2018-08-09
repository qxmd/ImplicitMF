#!/usr/bin/env python

"""
Unit tests for pre-processing module
====================================
"""

import pytest
import numpy as np

from implicitmf.preprocess import normalize_X


def test_normalize_X_output():
    """
    Check that output of normalize_X()
    is a scipy.sparse.csr matrix.
    """
    pass
    # normalize_X(X)

def test_normalize_X_incorrect_sparse_matrix():
    """
    Check that normalize_X() raises an
    error if X is not the correct format.
    """
    pass
    # normalize_X(X="hello")

def test_normalize_X_incorrect_norm_type():
    """
    Check that normalize_X() raises an error
    if norm_type is not one of bm25 or tfidf.
    """
    pass
