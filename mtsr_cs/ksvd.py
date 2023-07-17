import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import DictionaryLearning


class KSVD:
    def __init__(self, dictionary: np.ndarray, verbose=False):
        self.dictionary = dictionary
        self.code = None
        self.verbose = verbose

    def fit(self, X: np.ndarray):
        """
        X: shape (n, N_F)
        dictionary: shape (N_C, N_F)
        ----
        X: (n, N_F)
        SparseCoding: X(n, N_F) = Code(T, N_C)*Dict(N_C, N_F)
        In the paper: X(N_F, n) = psi(N_F, N_C)*S(N_C, n)
        --> self.dictionary = Dict(N_C, N_F) = (psi).T
        --> self.code = Code = (S)^T
        """
        n, F = X.shape
        D = self.dictionary
        N_C, N_F = D.shape
        assert N_C == N_F
        assert F == N_F
        init_code = np.zeros(shape=(n, N_C))
        dict_learner = DictionaryLearning(n_components=N_C, transform_algorithm='lasso_lars', random_state=42,
                                          fit_algorithm='cd', dict_init=D, positive_code=True, code_init=init_code,
                                          max_iter=10000, verbose=self.verbose)
        self.code = dict_learner.fit_transform(X)
        self.dictionary = np.array(dict_learner.components_)
        return self.dictionary, self.code
