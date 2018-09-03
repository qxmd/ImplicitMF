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
from skopt import forest_minimize
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k
from implicit.cuda import HAS_CUDA
from implicitmf.validation import cross_val_folds
from implicitmf._utils import _sparse_checker, _dict_checker
from implicitmf.preprocess import normalize_X
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
    the optimal hyperparemters of a model using `precision@10`
    as the evaluation metric.

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

    Note
    ----
    Adapted from sklearn's GridSearchCV for non-sklearn models.
    """
    _sparse_checker(X, '`X`')

    folds = cross_val_folds(X, n_folds=n_folds)

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
            X_train = folds[fold]['train']
            X_test = folds[fold]['test']
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

def _obj_als(hyperparams):
    folds = cross_val_folds(X, num_folds=n_folds)
    precision = []
    for fold in np.arange(n_folds):
        model = AlternatingLeastSquares(factors=hyperparams['regularization'],
                                regularization=hyperparams['factors'],
                                calculate_training_loss=True, use_gpu=HAS_CUDA)
        X_train = folds[fold]['train']
        X_test = folds[fold]['test']
        print("Fitting model...")
        model.fit(X_train.T)
        print("Calculating p@k...")
        recommendations = recommendations_sparse_array(model, np.arange(X.shape[0]), X_train, k=10, num_threads=n_threads)
        p = precision_at_k(X_test, recommendations, k=10)
        precision.append(p)
    # get mean p@k across all folds
    mean_precision = np.mean(precision)
    print(mean_precision)
    return -mean_precision

def _obj_ltr(hyperparams):
    """
    Calculates p at k for LightFM's model
    """
    folds = cross_val_folds(X, n_folds=n_folds)
    precision = []
    for fold in np.arange(n_folds):
        model = LightFM(loss='warp',
                learning_rate=hyperparams['learning_rate'],
                no_components=hyperparams['no_components'],
                user_alpha=hyperparams['user_alpha'],
                item_alpha=hyperparams['item_alpha'])
        X_train = folds[fold]['train']
        X_test = folds[fold]['test']
        print("Fitting model...")
        model.fit(X_train, epochs=10, num_threads=n_threads)
        print("Calculating p@k...")
        recommendations = recommendations_sparse_array(model, np.arange(X.shape[0]), X_train, k=10, num_threads=n_threads)
        p = precision_at_k(X_test, recommendations, k=10)
        precision.append(p)

    # get mean p@k across all folds
    mean_precision = np.mean(precision)
    print(mean_precision)
    return -mean_precision

def smbo(X, obj, model, hyperparams, n_threads, n_calls=100, n_jobs=1):
    """
    Performs sequential model-based optimization to
    identify optimal hyperparameters.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        utility matrix
    obj : func
        objective function that minimizes precision@k
    hyperparams : dict
        dictionary that specifies (min, max, interval) of
        each hyperparameter of interest
    n_threads : int
        number of threads to use in parallel

    Returns
    -------
    dict
        dictionary of optimal hyperparameters
    """
    _sparse_checker(X, '`X`')
    _dict_checker(hyperparams, '`hyperparams`')

    model_types = ['als', 'ltr']
    if model not in model_types:
        raise ValueError("`model` must be either 'ltr' or 'als'")

    if model == 'ltr':
        params = ['learning_rate', 'no_components', 'user_alpha', 'item_alpha']

    if model == 'als':
        params = ['regularization', 'factors']

    if hyperparams.keys() not in params:
        raise ValueError("`hyperparams` does not have the right keys")
    
    X = normalize_X(X, norm_type="bm25")
    res = forest_minimize(
        func=obj, dimensions=hyperparams, n_calls=n_calls, verbose=True, n_jobs=n_jobs)
    print('Maximum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))
    return dict(zip(params, res.x))
