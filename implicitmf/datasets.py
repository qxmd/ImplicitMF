#!/usr/bin/env python

import pandas as pd

def movielens(type='df'):
    """
    Gets a subset of data from MovieLens containing
    100,000 ratings by 700 users on 9,000 movies.

    Args
    ----
    type : str
        indicates output type of data, can be either
        'df' (dataframe) or 'array' 
    
    Returns
    -------
    pd.DataFrame or np.ndarray
        returns either dataframe or array of shape (n_ratings, 3)
    """
    path = "https://s3-us-west-2.amazonaws.com/implicitmf/movielens.csv"
    data = pd.read_csv(path)
    data = data.drop(columns=['timestamp'])
    data.columns = ['user_id', 'item_id', 'rating']
    return data

