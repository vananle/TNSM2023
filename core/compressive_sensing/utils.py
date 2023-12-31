import numpy as np

from .ksvd import KSVD
from core.utils.data_loading import load_raw_data


def get_psi(args, data):
    X = data['train/y_gt']
    nsamples = 500 if X.shape[0] > 500 else X.shape[0]
    X = X[-nsamples:]

    X = np.max(X, axis=1)
    # X_temp = X_temp[X_temp_max > 1.0]

    N_F = X.shape[1]
    D = np.zeros(shape=(N_F, N_F))

    psiT, ST = KSVD(D).fit(X)
    return psiT, ST


def get_phi(top_k_index, nseries):
    G = np.zeros((top_k_index.shape[0], nseries))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


def random_dictionary(n, K, normalized=True, seed=None):
    """
    Build a random dictionary matrix with K = n
    Args:
        n: square of signal dimension
        K: square of desired dictionary atoms
        normalized: If true, columns will be l2-normalized
        seed: Random seed

    Returns:
        Random dictionary
    """
    if seed:
        np.random.seed(seed)
    H = np.random.rand(n, K) * 255
    if normalized:
        for k in range(K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])
    return np.kron(H, H)


def dctii(v, normalized=True, sampling_factor=None):
    n = v.shape[0]
    K = sampling_factor if sampling_factor else n
    y = np.array([sum(np.multiply(v, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
    if normalized:
        y[0] = 1 / np.sqrt(2) * y[0]
        y = np.sqrt(2 / n) * y
    return y


def dictionary_from_transform(transform, n, K, normalized=True, inverse=True):
    """
    Builds a Dictionary matrix from a given transform
    Args:
        transform: A valid transform (e.g. Haar, DCT-II)
        n: number of rows transform dictionary
        K: number of columns transform dictionary
        normalized: If True, the columns will be l2-normalized
        inverse: Uses the inverse transform (as usually needed in applications)
    """
    H = np.zeros((K, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        H[:, i] = transform(v, sampling_factor=K)
    if inverse:
        H = H.T
    return np.kron(H.T, H.T)


def overcomplete_idctii_dictionary(n, K):
    """
    Build an overcomplete inverse DCT-II dictionary matrix with K > n
    Args:
        n: square of signal dimension
        K: square of desired number of atoms
    """
    if K > n:
        return dictionary_from_transform(dctii, n, K, inverse=False)
    else:
        raise ValueError("K needs to be larger than n.")


def unitary_idctii_dictionary(n):
    """
    Build a unitary inverse DCT-II dictionary matrix with K = n
    Args:
        n: square of signal dimension
    """
    return dictionary_from_transform(dctii, n, n, inverse=False)
