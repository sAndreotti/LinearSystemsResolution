from enum import Enum

import scipy as sp
import numpy as np

DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 20000

class InitializationMode(Enum):
    ZEROS = 0
    RANDOM = 1

DEFAULT_INITIALIZATION_MODE = InitializationMode.ZEROS
DEFAULT_LOWER_BOUND = -1
DEFAULT_UPPER_BOUND = 1

class ForwardSubstitutionMode(Enum):
    NAIVE = 0
    SCIPY = 1

DEFAULT_FORWARD_SUBSTITUTION_MODE = ForwardSubstitutionMode.SCIPY

def create_b(A:sp.sparse.csr_matrix, x:np.ndarray=None) -> np.ndarray:
    if x is None:
        x = np.ones(A.shape[1], dtype=np.float64)
        
    return A.dot(x)

