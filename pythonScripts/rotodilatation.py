import numpy as np

# Method for finding the rotation matrices that maps vector v0 to vector v1
# Based on the paper "N-dimensional Rotation Matrix Generation Algorithm"
def fnAR(v):
    n = np.size(v)
    R = np.eye(n)
    step = 1
    while step < n:
        A = np.eye(n)
        it = 0
        while it < n - step:
            r2 = (v[it] * v[it]) + (v[it + step] * v[it + step])
            if r2 > 0:
                r = np.sqrt(r2)
                pcos = v[it] / r
                psin = v[it + step] / r

                A[it, it] = pcos
                A[it, it + step] = -psin
                A[it + step, it] = psin
                A[it + step, it + step] = pcos
            it += 2 * step
        step *= 2
        v = np.dot(A, v)
        R = np.dot(A, R)
    return R

# Function computing the rotation matrix that maps vector v0 to vector v1
def compute_rotodilation(v0, v1):
    norm_v0 = np.linalg.norm(v0, 2)
    norm_v1 = np.linalg.norm(v1, 2)

    v0_normalized = v0 / norm_v0
    v1_normalized = v1 / norm_v1

    Mv0 = fnAR(v0_normalized)
    Mv1 = fnAR(v1_normalized)
    M = np.dot(np.transpose(Mv1), Mv0)
    M *= norm_v1 / norm_v0
    return M
