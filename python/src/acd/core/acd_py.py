import numpy as np
from acd.helpers import project_on_SO3

def acd_py(
    R_est,
    cost_matrix,
    constant_term,
    observed_indices,
    max_iters,
    eps_abs,
    eps_rel,
    print_frequency,
    shuffle_k,
):
    n = R_est.shape[0]

    converged = False
    obj_val = constant_term - np.sum(cost_matrix * RRT(R_est))

    if print_frequency != -1:
        print(f"iter #0: obj_val: {obj_val}")

    obj_val_next = np.inf
    for it in range(1, max_iters + 1):

        R_next = R_est.copy()

        order = np.random.permutation(n) if shuffle_k else np.arange(n)

        for k in order:
            obs = observed_indices[k]
            if len(obs) == 0:
                continue

            R_next[k] = project_on_SO3(bmmWTR(cost_matrix[k, obs], R_next[obs]))

        obj_val_next = constant_term - np.sum(cost_matrix * RRT(R_next))

        diff = abs(obj_val_next - obj_val)
        rel_diff = diff / max(abs(obj_val), 1.0)

        converged = (diff < eps_abs) or (rel_diff < eps_rel)

        if print_frequency != -1 and (it % print_frequency == 0 or converged):
            print(
                f"iter #{it}: obj_val: {obj_val_next}, delta_obj: {obj_val_next - obj_val}"
            )

        R_est = R_next
        obj_val = obj_val_next

        if converged:
            break
    return converged, R_est, obj_val


def bmmWTR(W, R):
    n = R.shape[0]

    Wm = np.transpose(W, (2, 1, 0)).reshape(3, 3 * n)
    Rm = np.transpose(R, (2, 1, 0)).reshape(3, 3 * n)

    return Wm @ Rm.T


def RRT(R):
    R = np.asarray(R)
    n = R.shape[0]

    RRT = np.zeros((n, n, 3, 3))

    for i in range(n):
        for j in range(n):
            RRT[i, j] = R[j] @ R[i].T

    return RRT