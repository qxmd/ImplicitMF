#!/usr/bin/env python

"""
Validation
==========
"""
import itertools
import copy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicitmf._utils import _sparse_checker
from implicit.evaluation import precision_at_k
from implicitmf._utils import _sparse_checker

def hold_out_entries(X, hold_out_size=0.2, seed=None):
    """
    Generates a sparse array of training examples by masking a random subset of values

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        sparse array of all observed interactions
    hold_out_size : float
        proportion of entries to be masked 
    seed : int
        random seed for use by np.random.choice

    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix of same shape as X with hold_out_size proportion of entries masked
    """
    _sparse_checker(X)

    # compute the number of nonzero entries in sparse array
    num_nonzero = X.count_nonzero()

    # set seed and randomly select some entries to be held out
    np.random.seed(seed)
    rand_hold_out = np.random.choice(np.arange(num_nonzero),
                                     size=int(
                                         np.floor(num_nonzero*hold_out_size)),
                                     replace=False)

    # get the indices of the nonzero components
    ind_nonzero = X.nonzero()

    # use randomly selected hold out values to pluck out corresponding indices
    indices_hold_out = (
        ind_nonzero[0][rand_hold_out], ind_nonzero[1][rand_hold_out])
    X[indices_hold_out] = 0
    X.eliminate_zeros()
    return X


def cross_val_folds(X, n_folds, seed=None):
    """
    Generates cross validation folds using provided utility matrix

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        utility matrix of shape (u, i) where u is number of users and i is number of items
    n_folds : int
        number of folds to create
    seed : int
        random seed for use by np.random.choice

    Returns
    -------
    dict
        dictionary of length n_folds
        
    Example
    -------
    >>> output = cross_val_folds(X, n_folds=3, seed=42)
    ... print(output)
    {0: {'train': X_train, 'test': X_test}, 
    1: {'train': X_train, 'test': X_test},
    2: {'train': X_train, 'test': X_test}}
    """
    _sparse_checker(X)

    if not isinstance(n_folds, int) or n_folds < 2:
        raise TypeError("`n_folds` must be an integer equal to or greater than 2")

    # compute the number of nonzero entries in sparse array
    num_nonzero = X.count_nonzero()
    ind_nonzero = X.nonzero()

    # set seed and shuffle the indices of the nonzero entries
    np.random.seed(seed)
    shuffled_ind = np.random.choice(
        np.arange(num_nonzero), size=num_nonzero, replace=False)

    fold_sizes = (num_nonzero // n_folds) * np.ones(n_folds, dtype=np.int)
    fold_sizes[:(num_nonzero % n_folds)] += 1

    split_shuffled_ind = dict()
    current = 0
    for key, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        split_shuffled_ind[key] = shuffled_ind[start:stop]
        current = stop

    # use the split shuffled indices to subset indices of nonzero entries from X
    val_indices = {key: (ind_nonzero[0][val], ind_nonzero[1][val])
                   for key, val in split_shuffled_ind.items()}

    folds = dict()
    for i in range(n_folds):
        print('Creating fold number {} ...'.format(i+1))
        test = csr_matrix((np.array(X[val_indices[i]]).reshape(-1),
                           val_indices[i]), shape=X.shape)

        train = X - test
        train.eliminate_zeros()

        folds[i] = {'train': train, 'test': test}
    return folds

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
    X : scipy.sparse.csr_matrix
    n_folds : int
        number of folds for cross-validation
    hyperparams : dict
        hyperparameter values of interest

    Returns
    -------
    pandas.DataFrame
        dataframe with mean_score, max_score, min_score for each combination of hyperparmeter values

    References
    ----------
    .. [1] scikit-learn's GridSearchCV: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
    """
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