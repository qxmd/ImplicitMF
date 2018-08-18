# ImplicitMF 

[![Build Status](https://travis-ci.org/qxmd/ImplicitMF.svg?branch=master)](https://travis-ci.org/qxmd/ImplicitMF)

ImplicitMF is a Python package that generates personalized recommendations for implicit feedback datasets. Unlike explicit feedback (e.g., movie ratings), implicit feedback looks at a user's interactions with an item and uses this as a surrogate measure of their preference toward that item. 

ImplicitMF provides a set of tools for building a recommendation system pipeline. These tools facilitate data pre-processing, hyperparameter training, model fitting, evaluation, validation, and post-processing of results. ImplicitMF focuses on two types of matrix factorization models that are built specifically for implicit feedback datasets:

1. **Alternating Least Squares (ALS):** as described in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). See [implicit](https://github.com/benfred/implicit) package for more information on its Python implementation.
2.  **Learning to Rank:** as described in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) and [WSABIE: Scaling Up To Large Vocabulary Image Annotation](https://research.google.com/pubs/archive/37180.pdf). See [LightFM](https://github.com/lyst/lightfm) package for more information on its Python implementation.

Further documentation can be found [here](https://qxmd.github.io/ImplicitMF/).

