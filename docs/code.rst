Transform
=========
.. autoclass:: implicitmf.transform.Transformer
    :members:

Pre-process
===========
.. autofunction:: implicitmf.preprocess.normalize_X

Validation
===========
In order to validate the performance of a recommender system,
we must first split the dataset, X, into X_train and X_validate. The traditional approach
to `train_test_split <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_ is to split dataset X either by row or column, thus resulting 
in a training set and validation set of different dimensions. However, in recommendation
systems, we perform train_test_split by "masking" a proportion of user-collection interactions
during the training phase then calculating precision@k by comparing predicted recommendations on
X_train against the original X matrix.

.. image:: imgs/train_test_split_2x.png
   :width: 500px
   :height: 180px
   :alt: alternate text
   :align: center


:py:func:`implicitmf.validation.cross_val_folds`
both use the "masked-out" approach to split data.

.. autofunction:: implicitmf.validation.cross_val_folds

Tune
====
.. autofunction:: implicitmf.tune.gridsearchCV
.. autofunction:: implicitmf.tune.smbo


Post-process
============
.. autofunction:: implicitmf.postprocess.remove_subscribed_items

