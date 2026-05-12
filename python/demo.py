from acd import acd
from acd.data import generate_data, read_matlab_data
from acd.helpers import rms, align_rotations, R2w
import numpy as np


def run_and_eval_acd(R_true, Rrel, H, anisotropic_cost=True):
    """
    Args:
        anisotropic_cost -- if true, uses the anisotropic cost
    """
    R_est, stat, stime, obj_val = acd(
        Rrel, H, anisotropic_cost=anisotropic_cost, print_frequency=10
    )

    # Evaluate the solution
    n = R_true.shape[0]
    R_est = align_rotations(R_est, R_true)
    fro_err = np.sqrt(np.sum((R_est - R_true) ** 2))
    angles = np.zeros(n)
    for i in range(n):
        angles[i] = np.linalg.norm(R2w(R_true[i] @ R_est[i].T)) / np.pi * 180
    angular_err = rms(angles)
    print(
        f"Frobenius error: {fro_err}, Angular error: {angular_err}, Solver runtime: {stime}"
    )


if __name__ == "__main__":
    print("Synthetic dataset:")
    n = 2000  # number of cameras
    p = 0.1  # proportion of observed relative rotations
    p_outlier = 0.5  # proportion of outlying relative rotations
    min_H_eigval = 1  # set to np.inf for noiseless data
    max_H_eigval = 100  # set to np.inf for noiseless data
    R_true, Rrel, H = generate_data(
        n, p, min_H_eigval=min_H_eigval, max_H_eigval=max_H_eigval, p_outlier=p_outlier
    )
    run_and_eval_acd(R_true, Rrel, H, anisotropic_cost=False)
    run_and_eval_acd(R_true, Rrel, H, anisotropic_cost=True)

    print("LU Sphinx dataset:")
    R_true, Rrel, H = read_matlab_data("../data/lu_sphinx.mat")
    run_and_eval_acd(R_true, Rrel, H, anisotropic_cost=False)
    run_and_eval_acd(R_true, Rrel, H, anisotropic_cost=True)
