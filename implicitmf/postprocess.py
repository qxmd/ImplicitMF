#!/usr/bin/env python

"""
Post-processing
===============
Operations to perform on recommendations output.
"""

from tqdm import tqdm
import numpy as np
from implicitmf._utils import _dict_checker

def remove_subscribed_items(rec_dict, user_sub_dict, unwanted_items=None):
    """
    Filters out already-subscribed collections from
    recommendations list for each user.

    Parameters
    ----------
    rec_dict : dict
        dictionary with user id as the key and list of recommended items as
        the value
    user_sub_dict : dict
        dictionary with user id as the key and list of item subscriptions as the value
    unwanted_items : list
        list of additional items to remove from the recommendation list
    
    Returns
    -------
    dict
        dictionary with recommended items that users have not subscribed to
    """
    _dict_checker(rec_dict, '`rec_dict`')
    _dict_checker(user_sub_dict, '`user_sub_dict`')

    if unwanted_items is not None:
        if not isinstance(unwanted_items, list):
            raise TypeError("unwanted_items variable must be a list")

    rec_set = set(rec_dict)
    user_sub_set = set(user_sub_dict)

    for i in tqdm(rec_set.intersection(user_sub_set)):
        d = dict(zip(rec_dict[i]['recs'],rec_dict[i]['score']))
        recommendations = rec_dict[i]['recs']
        subscribed = user_sub_dict[i]
        if unwanted_items:
            subscribed.append(unwanted_items)
        keep = set(recommendations)-set(subscribed)
        rec_dict[i]['recs'] = list(keep)
        rec_dict[i]['score'] = list({k: d[k] for k in list(keep)}.values())
    return rec_dict