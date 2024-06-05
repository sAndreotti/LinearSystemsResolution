from solver.Solver import Solver
from solver import Util

import numpy as np
import scipy as sp

class ConjugateGradientSolver(Solver): 
    '''
    Implementation of the Conjugate Gradient resolution method. 
    '''

    def __init__(self, tol:float=Util.DEFAULT_TOL, max_iter:int=Util.DEFAULT_MAX_ITER, 
                 initialization_mode:Util.InitializationMode=Util.DEFAULT_INITIALIZATION_MODE,
                 lower_bound:float=Util.DEFAULT_LOWER_BOUND, upper_bound:float=Util.DEFAULT_UPPER_BOUND):
        
        super().__init__(tol=tol, max_iter=max_iter, initialization_mode=initialization_mode,
                         lower_bound=lower_bound, upper_bound=upper_bound)
    
    def _update_x(self, A:sp.sparse.csr_matrix, b:np.ndarray, x:np.ndarray,
                  d:np.array) -> tuple[np.array, np.array, np.array]:
        r = b - A * x
        y = A * d 

        alpha = (d @ r) / (d @ y)
        x = x + alpha * d

        r = b - A * x
        w = A * r
        beta = (d @ w) / (d @ y)
        d = r - beta * d
        
        return x, r, d
    
    def solve(self, A:sp.sparse.csr_matrix, b:np.ndarray) -> np.ndarray:
        '''
        Solve the system of equations Ax = b using the Conjugate Gradient method.
        
        Parameters:
        @param A: sp.sparse.csr_matrix -> Matrix of the system.
        @param b: np.ndarray -> Known terms vector of the system.
        
        @return x: np.ndarray -> Solution of the system.
        '''

        x_0 = super()._initialize_x_0(A.shape[1])
        return super().solve(A, b, support= b - A * x_0)