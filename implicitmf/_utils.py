from scipy.sparse import csr_matrix

def _sparse_checker(X):
    if not isinstance(X, csr_matrix):
        raise TypeError("`X` must be a scipy.sparse.csr_matrix")

def _dict_checker(input_dict, var_name):
    if not isinstance(input_dict, dict):
        raise TypeError("{:s} must be a dict".format(var_name))