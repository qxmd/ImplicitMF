#!/usr/bin/env python

"""
Transformer
===========
"""

import numpy as np
from scipy.sparse import csr_matrix

class Transformer(object):
    """
    Transform fetched results into sparse matrix.

    Parameters
    ----------
    user_item_dict : dict
        a dictionary of lists of tuples containing distinct pairs of ids,
        distinct user ids, and distinct item ids

    Attributes
    ----------
    user_item_score : list
        list of tuples of length three, where first item in tuple is user_id
        second is item_id, third is score
    user_mapper : dict
        keys are user_ids and values are indices along user axis in user item matrix
    item_mapper : dict
        keys are item_ids and values are indices along item axis in user item matrix
    user_inv_mapper : dict
        keys are indices along user axis in user item matrix and values are user_ids
    item_inv_mapper : dict
        keys are indices along item axis in user item matrix and values are item_ids
    """
    def __init__(self, user_item_dict):
        if not isinstance(user_item_dict, dict):
            raise TypeError("`user_item_dict` must be a dict")

        self.user_item_score = user_item_dict['user_item_score']
        self.distinct_user_ids = user_item_dict['user_id']
        self.distinct_item_ids = user_item_dict['item_id']

        self.user_mapper = dict(zip(self.distinct_user_ids, range(len(self.distinct_user_ids))))
        self.item_mapper = dict(zip(self.distinct_item_ids, range(len(self.distinct_item_ids))))
        self.user_inv_mapper = dict(enumerate(self.distinct_user_ids))
        self.item_inv_mapper = dict(enumerate(self.distinct_item_ids))

    def to_sparse_array(self, arr_type='csr_matrix'):
        """
        Transforms provided data into scipy sparse array

        Parameters
        ----------
        type : str
            a string indicating type of sparse array returned (only supports csr_matrix)

        Returns
        -------
        scipy.sparse.csr_matrix
            utility matrix of shape (u,i) where u represents number of distinct users and
            i represents number of distinct items
        """
        supported_types = ['csr_matrix']

        if arr_type not in supported_types:
            raise TypeError("`arr_type` must be a csr_matrix")

        user_bool = np.isin(self.user_item_score[:,0], self.distinct_user_ids)
        item_bool = np.isin(self.user_item_score[:,1], self.distinct_item_ids)

        keep = np.logical_and(user_bool, item_bool)

        lost_int = len(keep) - np.sum(keep)
        print('{} interactions removed by filtered distinct user/item ids'.format(lost_int))

        sparse_fill = self.user_item_score[keep]

        for i in range(sparse_fill.shape[0]):
            sparse_fill[i,0] = self.user_mapper[sparse_fill[i,0]]
            sparse_fill[i,1] = self.item_mapper[sparse_fill[i,1]]

        if arr_type == 'csr_matrix':
            arr = csr_matrix((sparse_fill[:,2], (sparse_fill[:,0], sparse_fill[:,1])),
                             shape=(len(self.user_mapper),
                                    len(self.item_mapper)))
        return arr