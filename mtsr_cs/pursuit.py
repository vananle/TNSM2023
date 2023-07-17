import os

import numpy as np
from sklearn.decomposition import SparseCoder


def sparse_coding(ZT, phiT, psiT):
    """
    As SparseCoding: X = Code*Dict | As in the paper: Zt = phi*psi*S
    X:(n, N_F); Code:(n, N_C); Dict:(N_C, N_F)
    --> Z.T = S.T * psi.T * phi.T
    Z.T:(n, k); S.T:(n, N_C) | psi.T:(N_C, N_F)=Dict, phi.T:(N_F, k)
    k: number of topk flows
    N_C=N_F: total number of flows
    n: total number timesteps/samples

    ------
    Input:
    - ZT:(n, k)
    - phiT:(N_F, k)
    - psiT: (N_C, N_F)
    return:
    - Shat:(n, N_C)
    """
    # analyze shape of Y
    if len(ZT.shape) == 1:
        data = np.array([ZT])
    elif len(ZT.shape) == 2:
        data = np.copy(ZT)
    else:
        raise ValueError("Input must be a vector or a matrix.")

    # analyze dimensions
    N_C, N_F = psiT.shape
    assert N_C == N_F
    k = phiT.shape[1]
    A = np.dot(psiT, phiT)  # shape (N_C, k)
    assert k == ZT.shape[1]

    coder = SparseCoder(dictionary=A, transform_algorithm='lasso_lars',
                        transform_alpha=1e-10, positive_code=True, n_jobs=os.cpu_count(),
                        transform_max_iter=10000)

    Shat = coder.transform(ZT)
    return Shat  # shape(n, N_C)
