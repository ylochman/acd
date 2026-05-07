import numpy as np
from scipy.linalg import svd, expm
import time

from .helpers import skewsym, normalize
from .core import acd_core


def acd(
    Rrel,
    H,
    anisotropic_cost=True,
    init="zero",
    shuffle_k=True,
    max_iters=1000,
    eps_abs=1e-12,
    eps_rel=1e-12,
    print_frequency=-1,
):

    cost_matrix, constant_term, observed_indices = construct_cost_matrix(
        Rrel, H, anisotropic=anisotropic_cost
    )

    n = Rrel.shape[0]

    start = time.time()

    R_est = initialize_rotations(n, init).copy()

    converged, R_est, obj_val = acd_core(
        R_est,
        cost_matrix,
        constant_term,
        observed_indices,
        max_iters,
        eps_abs,
        eps_rel,
        print_frequency,
        shuffle_k,
    )

    runtime = time.time() - start

    stat = "converged" if converged else "reached the maximum number of iterations"

    return R_est, stat, runtime, obj_val


def construct_cost_matrix(Rrel, H, anisotropic=True):
    n = Rrel.shape[0]
    I3 = np.eye(3)

    k_observed = 0
    constant_term = 0.0

    cost_matrix = np.zeros((n, n, 3, 3))
    observed_indices = [[] for _ in range(n)]

    # detect observed edges
    for i in range(n):
        observed_indices[i] = [
            x for x in range(n) if x != i and np.sum(Rrel[x, i] ** 2) > 0
        ]

        for j in observed_indices[i]:
            if j > i:
                k_observed += 1

                if anisotropic:
                    H_ij = H[i, j]
                    M_ij = (np.trace(H_ij) / 2.0) * I3 - H_ij

                    cost_matrix[i, j] = M_ij @ Rrel[i, j]
                    constant_term += 2.0 * np.trace(M_ij)
                else:
                    cost_matrix[i, j] = Rrel[i, j]
                    constant_term += 6.0

                cost_matrix[j, i] = cost_matrix[i, j].T

    k_observed *= 2

    return cost_matrix / k_observed, constant_term / k_observed, observed_indices


def initialize_rotations(n, init, max_axis_angle_norm=360):
    R = np.zeros((n, 3, 3))

    if init == "id":
        R[:] = np.tile(np.eye(3), (n, 1, 1))

    elif init in ["randn", "svd", "axis_angle"]:
        R[:] = np.random.randn(n, 3, 3)

        if init in ["svd", "axis_angle"]:
            max_axis_angle_norm *= 2 * np.pi / 360

            for i in range(n):
                if init == "svd":
                    U, _, Vt = svd(R[i])
                    R[i] = U @ Vt
                else:
                    axis = normalize(np.random.rand(3))
                    angle = np.random.rand() * max_axis_angle_norm
                    R[i] = expm(skewsym(axis * angle))

    return R
