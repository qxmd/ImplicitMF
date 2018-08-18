#!/usr/bin/env python

"""
Unit tests for tune module
===========================
"""

import pytest
import pandas as pd
from implicitmf.tune import  gridsearchCV
from implicit.als import AlternatingLeastSquares
from _mock_data import sparse_array


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