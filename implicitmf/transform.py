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
    item_sub_dict: dict
        a dictionary of lists of tuples containing distinct pairs of ids,
        distinct user ids, and distinct item ids
    full_matrix: boolean
        Default is True. Determines whether matrix will be an 
        "out matrix" or "in matrix".

    Attributes
    ----------
    user_item_score: list
        list of tuples of length three, where first item in tuple is user_id
        second is item_id, third is score
    user_mapper: dict
        keys are user_ids and values are indices along user axis in user item matrix
    item_mapper: dict
        keys are item_ids and values are indices along item axis in user item matrix
    user_inv_mapper: dict
        keys are indices along user axis in user item matrix and values are user_ids
    item_inv_mapper: dict
        keys are indices along item axis in user item matrix and values are item_ids
    Examples
    --------
    >>> from implicitmf.transform import Transformer
    >>> uc_dict, _ = gen_fetched_data()
    >>> t = Transformer(uc_dict)
    >>> X = t.to_sparse_array(arr_type='csr_matrix')
    """

    def __init__(self, user_item_dict, full_matrix=True):
        if not isinstance(full_matrix, bool):
            raise ValueError("full_matrix parameter must be a boolean.")
        if not isinstance(user_item_dict, dict):
            raise ValueError("user_item_dict parameter must be a dict.")

        self.user_item_score = user_item_dict['item_user_score']
        
        if full_matrix is True:
            self.distinct_user_ids = user_item_dict['user_id']
            self.distinct_item_ids = user_item_dict['item_id']
        else:
            self.distinct_user_ids = np.array(list(set(self.user_item_score[:,0]).intersection(user_item_dict['user_id'])))
            self.distinct_item_ids = np.array(list(set(self.user_item_score[:,1]).intersection(user_item_dict['item_id'])))

            diff_users =  len(user_item_dict['user_id']) - len(self.distinct_user_ids)
            diff_items =  len(user_item_dict['item_id']) - len(self.distinct_item_ids)

            print('{} users removed by `full_matrix` option'.format(diff_users))
            print('{} items removed by `full_matrix` option'.format(diff_items))

        self.user_mapper = dict(zip(self.distinct_user_ids, range(len(self.distinct_user_ids))))
        self.item_mapper = dict(zip(self.distinct_item_ids, range(len(self.distinct_item_ids))))
        self.user_inv_mapper = dict(enumerate(self.distinct_user_ids))
        self.item_inv_mapper = dict(enumerate(self.distinct_item_ids))

    def to_sparse_array(self, arr_type='csr_matrix'):
        """
        Transforms provided data into scipy sparse array
        Parameters
        ----------
        type: str
            a string indicating type of sparse array returned (only supports csr_matrix)
        Returns
        -------
        scipy.sparse_csr_matrix
            Scipy sparse array of desired type
        """

        supported_types = ['csr_matrix']

        if arr_type not in supported_types:
            raise TypeError('That array type is not supprted')

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