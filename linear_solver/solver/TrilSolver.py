import numpy as np
import scipy as sp
import scipy.linalg

from solver.Util import ForwardSubstitutionMode, DEFAULT_FORWARD_SUBSTITUTION_MODE

class TrilSolver():
    '''
    Class to resolve lower triangular systems using the forward substituion.
    '''

    def __init__(self, mode:ForwardSubstitutionMode=DEFAULT_FORWARD_SUBSTITUTION_MODE):
        '''
        Constructor of the class TrilSolver.

        Parameters:
        @A: sp.sparse.csr_matrix -> Lower triangular matrix.
        '''
        self.mode = mode

    def forward_substitution(self, L:sp.sparse.csr_matrix, b:np.array) -> np.array:
        '''
        Function to resolve the lower triangular system Lx = b using the forward substitution.

        Parameters:
        @L: sp.sparse.csr_matrix -> Lower triangular matrix.
        @b: np.array -> Vector b.

        Returns:
        @x: np.array -> Solution of the system.
        '''
        if self.mode == ForwardSubstitutionMode.NAIVE:
            return self._forward_substitution_naive(L, b)
        elif self.mode == ForwardSubstitutionMode.SCIPY:
            return self._forward_substitution_scipy(L, b)

        raise Exception("Forward substitution mode not supported")
    
    def _forward_substitution_naive(self, L:sp.sparse.csr_matrix, b:np.array) -> np.array:
        if not np.all(L.diagonal()):
            raise ValueError("Matrix L has zeros on the diagonal: forward substitution not applicable.")
        
        n = L.shape[0]
        x = np.zeros(n, dtype=np.float64)

        x[0] = b[0] / L[0, 0]
        
        for i in range(1, n):
            x[i] = (b[i] - L[i, :i].toarray().flatten().dot(x[:i])) / L[i, i]  

        return x
    
    def _forward_substitution_scipy(self, L:sp.sparse.csr_matrix, b:np.array) -> np.array: 
        return scipy.linalg.solve_triangular(L.toarray(), b, lower=True)
