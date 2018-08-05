# ImplicitMF 

ImplicitMF is a Python package that generates personalized recommendations for implicit feedback datasets. Unlike explicit feedback (e.g., movie ratings), implicit feedback looks at a user's interactions with an item and uses this as a surrogate measure of their preference toward that item. 

ImplicitMF provides a selection of implicit-specific matrix factorization techniques to generate item recommendations for a set of users:

- **Alternating Least Squares (ALS):** as described in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). See [implicit](https://github.com/benfred/implicit) package for more information on its Python implementation.
-  **Learning to Rank:** as described in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) and [WSABIE: Scaling Up To Large Vocabulary Image Annotation](https://research.google.com/pubs/archive/37180.pdf). See [LightFM](https://github.com/lyst/lightfm) package for more information on its Python implementation.