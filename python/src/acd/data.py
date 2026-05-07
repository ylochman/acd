import numpy as np
from scipy.linalg import expm
from scipy.io import loadmat
from .helpers import skewsym, normalize

def generate_data(n, p, min_H_eigval=10, max_H_eigval=100, p_outlier=0.5):
    """
    Create synthetic camera graph.

    Args:
        n: number of cameras
        p: fraction of observed relative rotations
        min_H_eigval / max_H_eigval: eigenvalue range of precision matrices
        p_outlier: fraction of outliers

    Returns:
        R: (n,3,3) absolute rotations
        Rrel: (n,n,3,3) relative rotations
        H: (n,n,3,3) Hessians
    """
    k = n * (n - 1) // 2
    k_observed = max(n - 1, int(np.ceil(k * p)))
    k_outliers = int(np.ceil(k_observed * p_outlier))

    # Upper-triangular indices (i < j)
    triu_indices = np.array([(i, j) for i in range(n) for j in range(i+1, n)])

    success = False
    for _ in range(100):
        W = np.zeros((n, n), dtype=bool)
        O = np.zeros((n, n), dtype=bool)

        perm_idx = np.random.permutation(k)

        obs_pairs = triu_indices[perm_idx[:k_observed]]
        out_pairs = triu_indices[perm_idx[:k_outliers]]

        for i, j in obs_pairs:
            W[i, j] = True
        for i, j in out_pairs:
            O[i, j] = True

        W = W | W.T
        O = O | O.T

        if np.all(np.sum(W & ~O, axis=0) > 0):
            success = True
            break

    if not success:
        raise ValueError("Failed to generate a valid observation pattern after 100 attempts.")

    # Absolute rotations
    R = np.zeros((n, 3, 3))
    for i in range(n):
        axis = normalize(np.random.rand(3))
        angle = np.random.rand() * 2 * np.pi
        R[i] = expm(skewsym(axis * angle))

    # Relative rotations and Hessians
    Rrel = np.zeros((n, n, 3, 3))
    H = np.zeros((n, n, 3, 3))

    for i in range(n):
        for j in range(i+1, n):
            if W[i, j]:
                if O[i, j]:
                    # Outlier
                    axis = normalize(np.random.rand(3))
                    angle = np.random.rand() * 2 * np.pi
                    Rrel[i, j] = expm(skewsym(axis * angle))

                    eigvecs = expm(skewsym(normalize(np.random.rand(3)) * np.random.rand() * 2 * np.pi))
                    eigvals = np.random.rand(3) * 0.9 + 0.1
                    H[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T

                else:
                    # Inlier
                    Rrel_ij = R[j] @ R[i].T

                    if np.isinf(min_H_eigval) and np.isinf(max_H_eigval):
                        H[i, j] = np.eye(3)
                        Rrel[i, j] = Rrel_ij
                    else:
                        eigvecs = expm(skewsym(normalize(np.random.rand(3)) * np.random.rand() * 2 * np.pi))
                        eigvals = np.random.rand(3) * (max_H_eigval - min_H_eigval) + min_H_eigval
                        H[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T

                        dw = eigvecs @ (np.random.randn(3) / np.sqrt(eigvals))
                        Rrel[i, j] = expm(skewsym(dw)) @ Rrel_ij

                Rrel[j, i] = Rrel[i, j].T
                H[j, i] = H[i, j].T

    return R, Rrel, H


def read_matlab_data(matfile):
    data = loadmat(matfile)
    
    Rgt = data["Rgt"]
    Rrel_mat = data["Rrel"]
    H_mat = data["H"]
    
    n = Rgt.shape[0] // 3  # number of cameras
    
    # Absolute rotations
    R_true = np.zeros((n, 3, 3))
    for i in range(n):
        R_true[i] = Rgt[3*i:3*i+3, :]
    
    # Relative rotations and Hessians
    Rrel = np.zeros((n, n, 3, 3))
    H = np.zeros((n, n, 3, 3))
    
    for i in range(n):
        for j in range(i+1, n):
            Rrel[i, j] = Rrel_mat[3*i:3*i+3, 3*j:3*j+3]
            Rrel[j, i] = Rrel[i, j].T
            
            H[i, j] = H_mat[3*i:3*i+3, 3*j:3*j+3]
            H[j, i] = H[i, j].T
    
    return R_true, Rrel, H