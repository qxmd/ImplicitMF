#!/usr/bin/env python

import pandas as pd

def movielens():
    """
    Gets a subset of data from MovieLens containing
    100,000 ratings by 700 users on 9,000 movies.
    """
    path = "https://s3-us-west-2.amazonaws.com/implicitmf/movielens.csv"
    data = pd.read_csv(path)
    return data

