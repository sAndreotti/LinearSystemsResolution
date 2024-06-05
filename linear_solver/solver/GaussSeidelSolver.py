from solver.Solver import Solver
from solver import Util
from solver.TrilSolver import TrilSolver

import numpy as np
import scipy as sp

class GaussSeidelSolver(Solver): 
    '''
    Implementation of the Gauss-Seidel resolution method. 
    '''

    def __init__(self, tol:float=Util.DEFAULT_TOL, max_iter:int=Util.DEFAULT_MAX_ITER, 
                 initialization_mode:Util.InitializationMode=Util.DEFAULT_INITIALIZATION_MODE,
                 lower_bound:float=Util.DEFAULT_LOWER_BOUND, upper_bound:float=Util.DEFAULT_UPPER_BOUND,
                 forward_substitution_mode:Util.ForwardSubstitutionMode=Util.DEFAULT_FORWARD_SUBSTITUTION_MODE):
        
        super().__init__(tol=tol, max_iter=max_iter, initialization_mode=initialization_mode,
                         lower_bound=lower_bound, upper_bound=upper_bound)
        
        self._tril_solver = TrilSolver(mode=forward_substitution_mode)
        
    def _update_x(self, A:sp.sparse.csr_matrix, b:np.ndarray, x:np.ndarray,
                        P:sp.sparse.csr_matrix) -> tuple[np.array, np.array, sp.sparse.csr_matrix]:
    
        r = b - A.dot(x)
        y = self._tril_solver.forward_substitution(P, r)
        x = x + y

        return x, r, P
    
    def solve(self, A:sp.sparse.csr_matrix, b:np.ndarray) -> np.ndarray:
        '''
        Solve the system of equations Ax = b using the Gauss-Seidel method.
        
        Parameters:
        @A: sp.sparse.csr_matrix -> Matrix of the system.
        @b: np.ndarray -> Known terms vector of the system.
        
        Return:
        @x: np.ndarray -> Solution of the system.
        '''
        
        return super().solve(A, b, support=sp.sparse.tril(A, format='csr'))