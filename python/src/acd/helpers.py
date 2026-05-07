import numpy as np
from scipy.linalg import logm

def skewsym(t):
    t = np.asarray(t)
    return np.array([
        [0,     -t[2],  t[1]],
        [t[2],   0,    -t[0]],
        [-t[1],  t[0],  0]
    ])

def R2w(R):
    logR = logm(R)
    return np.array([logR[2, 1], logR[0, 2], logR[1, 0]])

def project_on_SO3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        D = np.diag([1, 1, -1])
        R = U @ D @ Vt
    return R

def rms(diff):
    diff = np.asarray(diff)
    return np.sqrt(np.mean(diff.ravel()**2))

def normalize(x, p=2):
    x = np.asarray(x)
    return x / (np.sum(x**p)**(1.0/p))


def align_rotations(R1, R2):
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    
    n = R1.shape[0]
    M_align = np.zeros((3, 3))
    
    for i in range(n):
        M_align += R1[i].T @ R2[i]
    
    R_align = project_on_SO3(M_align)
    
    for i in range(n):
        R1[i] = R1[i] @ R_align
    
    return R1