from solver.Solver import Solver
from solver import Util

import numpy as np
import scipy as sp

class JacobiSolver(Solver): 
    '''
    Implementation of the Jacobi resolution method. 
    '''

    def __init__(self, tol:float=Util.DEFAULT_TOL, max_iter:int=Util.DEFAULT_MAX_ITER, 
                 initialization_mode:Util.InitializationMode=Util.DEFAULT_INITIALIZATION_MODE,
                 lower_bound:float=Util.DEFAULT_LOWER_BOUND, upper_bound:float=Util.DEFAULT_UPPER_BOUND):
        
        super().__init__(tol=tol, max_iter=max_iter, initialization_mode=initialization_mode,
                         lower_bound=lower_bound, upper_bound=upper_bound)
    
    def _update_x(self, A:sp.sparse.csr_matrix, b:np.ndarray, x:np.ndarray, 
                  P_inv:sp.sparse.csr_matrix) -> tuple[np.array, np.array, sp.sparse.csr_matrix]:
    
        r = b - A.dot(x)
        x = x + P_inv.dot(r)

        return x, r, P_inv
    
    def solve(self, A:sp.sparse.csr_matrix, b:np.ndarray) -> np.ndarray:
        '''
        Solve the system of equations Ax = b using the Jacobi method.
        
        Parameters:
        @param A: sp.sparse.csr_matrix -> Matrix of the system.
        @param b: np.ndarray -> Known terms vector of the system.
        
        @return x: np.ndarray -> Solution of the system.
        '''
        
        return super().solve(A, b, support=sp.sparse.diags(1 / A.diagonal(), format='csr'))