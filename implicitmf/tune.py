#!/usr/bin/env python

"""
Hyperparameter Tuning
=====================
"""
import itertools
import copy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM, evaluation
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from implicit.cuda import HAS_CUDA
from implicitmf.validation import cross_val_folds
from implicitmf._utils import _sparse_checker

def _get_precision(model, X_train, X_test, K=10):
    """
    Gets precision@k of an Implicit ALS model

    Parameters
    ----------
    model : model object
    X_train : scipy.sparse.csr_matrix
        training set
    X_test : scipy.sparse.csr_matrix
        test set
    K : int
        number of recommendations to consider in precision@k 
    
    Returns
    -------
    float
        precision@k 
    """
    print("Fitting model...")
    model.fit(X_train.T)
    test_precision = precision_at_k(model, X_train, X_test, K=K)
    print("p@{:d}:".format(K), test_precision)
    return test_precision

def gridsearchCV(base_model, X, n_folds, hyperparams):
    """
    Performs exhaustive gridsearch cross-validation to identify
    the optimal hyperparemters of a model.

    Parameters
    ----------
    base_model : model object
        base model
    X : scipy.sparse.csr_matrix
        utility matrix
    n_folds : int
        number of folds for cross-validation
    hyperparams : dict
        hyperparameter values of interest

    Returns
    -------
    pandas.DataFrame
        dataframe with ``mean_score``, ``max_score``, ``min_score`` for each combination of hyperparmeter values

    References
    ----------
    .. [1] scikit-learn's GridSearchCV: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
    """
    _sparse_checker(X)

    fold_dict = cross_val_folds(X, n_folds=n_folds)

    keys, values = zip(*hyperparams.items())
    p_total = []
    hyperparam_vals = []
    max_score, min_score, mean_score = [], [], []
    df = pd.DataFrame(columns=list(keys))
    for val in itertools.product(*values):
        params = dict(zip(keys, val))
        this_model = copy.deepcopy(base_model)
        print_line = []
        for k, v in params.items():
            setattr(this_model, k, v)
            print_line.append((k, v))
        print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
        precision = []
        for fold in np.arange(n_folds):
            X_train = fold_dict[fold]['train']
            X_test = fold_dict[fold]['test']
            p = _get_precision(this_model, X_train, X_test, K=10)
            precision.append(p)
        p_total.append(precision)
        hyperparam_vals.append(list(val))
        max_score.append(max(precision))
        min_score.append(min(precision))
        mean_score.append(np.mean(precision))

    results = pd.DataFrame(hyperparam_vals)
    results['mean_score'] = mean_score
    results['max_score'] = max_score
    results['min_score'] = min_score
    return results

def smbo(X, obj, hyperparameters, n_threads, n_calls=100, n_jobs=1):
    """
    Performs sequential model-based optimization to
    identify optimal hyperparameters.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        utility matrix
    obj : func
        objective function that minimizes precision@k
    hyperparamers : list
        list of tuples that specify (min, max, interval) of
        each hyperparameter of interest
    n_threads : int
        number of threads to use in parallel

    Returns
    -------
    dict
        dictionary of optimal hyperparameters
    """
    pass
