import numpy as np


def outer(u, v, w):
    return np.einsum('i,j,k->ijk', u, v, w, dtype=np.int32, casting="same_kind")

def is_zero_tensor(tensor):
    return np.all(tensor == 0)