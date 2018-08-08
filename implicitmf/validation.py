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
from implicit.evaluation import precision_at_k


def hold_out_entries(sparse_arr, hold_out_size=0.2, seed=None):
    """
    Generates a sparse array of training examples by masking a random subset of values

    Parameters
    ----------
    sparse_arr: csr_matrix
        sparse array of all observed interactions
    hold_out_size: float
        proportion of entries to be masked 
    seed: int
        random seed for use by np.random.choice

    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix of same shape as sparse_arr with hold_out_size proportion of entries masked
    """
    if not isinstance(sparse_arr, csr_matrix):
        raise TypeError("`sparse_arr` must be a scipy sparse csr matrix")

    # compute the number of nonzero entries in sparse array
    num_nonzero = sparse_arr.count_nonzero()

    # set seed and randomly select some entries to be held out
    np.random.seed(seed)
    rand_hold_out = np.random.choice(np.arange(num_nonzero),
                                     size=int(
                                         np.floor(num_nonzero*hold_out_size)),
                                     replace=False)

    # get the indices of the nonzero components
    ind_nonzero = sparse_arr.nonzero()

    # use randomly selected hold out values to pluck out corresponding indices
    indices_hold_out = (
        ind_nonzero[0][rand_hold_out], ind_nonzero[1][rand_hold_out])
    sparse_arr[indices_hold_out] = 0
    sparse_arr.eliminate_zeros()
    return sparse_arr


def cross_val_folds(sparse_arr, num_folds, seed=None):
    """
    Generates cross validation folds using provided data

    Parameters
    ----------
    sparse_arr: csr_matrix
        sparse array of all observed interactions
    num_folds: int
        number of folds to create
    seed: int
        random seed for use by np.random.choice

    Returns
    -------
    dict
        dictionary of length num_folds of the form
        {0: {'train': X_train, 'test': X_test}, 1: ...}
    """

    if not isinstance(sparse_arr, csr_matrix):
        raise TypeError("`sparse_arr` must be a scipy sparse csr matrix")

    if not isinstance(num_folds, int) or num_folds < 2:
        raise TypeError("`num_folds` must be an int > 2")

    # compute the number of nonzero entries in sparse array
    num_nonzero = sparse_arr.count_nonzero()
    ind_nonzero = sparse_arr.nonzero()

    # set seed and shuffle the indices of the nonzero entries
    np.random.seed(seed)
    shuffled_ind = np.random.choice(
        np.arange(num_nonzero), size=num_nonzero, replace=False)

    fold_sizes = (num_nonzero // num_folds) * np.ones(num_folds, dtype=np.int)
    fold_sizes[:(num_nonzero % num_folds)] += 1

    split_shuffled_ind = dict()
    current = 0
    for key, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        split_shuffled_ind[key] = shuffled_ind[start:stop]
        current = stop

    # use the split shuffled indices to subset indices of nonzero entries from sparse_arr
    val_indices = {key: (ind_nonzero[0][val], ind_nonzero[1][val])
                   for key, val in split_shuffled_ind.items()}

    folds = dict()
    for i in range(num_folds):
        print('Creating fold number {} ...'.format(i+1))
        test = csr_matrix((np.array(sparse_arr[val_indices[i]]).reshape(-1),
                           val_indices[i]), shape=sparse_arr.shape)

        train = sparse_arr - test
        train.eliminate_zeros()

        folds[i] = {'train': train, 'test': test}
    return folds

def _get_precision(model, X_train, X_test, K=10):
    """
    Gets precision@k of an Implicit ALS model.

    Parameters
    ----------
    model: model object
    X_train: scipy.sparse.csr_matrix
        training set
    X_test: scipy.sparse.csr_matrix
        test set
    K: int
        k recommendations to consider in p@k 
    
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
    the best hyperparemters of a model.

    Parameters
    ----------
    base_model: model object
    X: scipy.sparse.csr_matrix
    n_folds: int
        number of folds for cross-validation
    hyperparams: dict
        hyperparameter values of interest

    Returns
    -------
    dataframe
        pandas dataframe with mean_score, max_score, min_score for each combination of hyperparmeter values

    References
    ----------
    .. [1] scikit-learn's GridSearchCV: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
    """
    fold_dict = cross_val_folds(X, num_folds=n_folds)

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