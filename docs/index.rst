.. ImplicitMF documentation master file, created by
   sphinx-quickstart on Thu Aug  9 15:58:41 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ImplicitMF's documentation!
======================================

ImplicitMF is a Python package that generates personalized recommendations for implicit feedback datasets. Unlike explicit feedback (e.g., movie ratings), implicit feedback looks at a user's interactions with an item and uses this as a surrogate measure of their preference toward that item. 

ImplicitMF provides a selection of implicit-specific matrix factorization techniques to generate item recommendations for a set of users:

- **Alternating Least Squares:** as described in `Collaborative Filtering for Implicit Feedback Datasets <http://yifanhu.net/PUB/cf.pdf//>`_ See `implicit <https://github.com/benfred/implicit//>`_ package for more information on its Python 
- **Learning to Rank:** as described in `BPR: Bayesian Personalized Ranking from Implicit Feedback <https://arxiv.org/pdf/1205.2618.pdf/>`_ and `WSABIE: Scaling Up To Large Vocabulary Image Annotation <https://research.google.com/pubs/archive/37180.pdf///>`_. See `LightFM <https://github.com/lyst/lightfm/>`_ package for more information on its Python implementation.

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation:

   code.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
