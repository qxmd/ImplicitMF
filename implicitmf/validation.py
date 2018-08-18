#!/usr/bin/env python

"""
Validation
==========
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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