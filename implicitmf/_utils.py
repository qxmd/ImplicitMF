from scipy.sparse import csr_matrix

def _sparse_checker(X):
    if not isinstance(X, csr_matrix):
        raise TypeError("X must be a sparse matrix of type csr.")
