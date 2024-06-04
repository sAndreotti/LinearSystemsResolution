from solver.Solver import Solver
from solver import Util

import numpy as np
import scipy as sp

class GradientSolver(Solver): 
    '''
    Implementation of the Gradient resolution method. 
    '''

    def __init__(self, tol:float=Util.DEFAULT_TOL, max_iter:int=Util.DEFAULT_MAX_ITER, 
                 initialization_mode:Util.InitializationMode=Util.DEFAULT_INITIALIZATION_MODE,
                 lower_bound:float=Util.DEFAULT_LOWER_BOUND, upper_bound:float=Util.DEFAULT_UPPER_BOUND):
        
        super().__init__(tol=tol, max_iter=max_iter, initialization_mode=initialization_mode,
                         lower_bound=lower_bound, upper_bound=upper_bound)
    
    def _update_x(self, A:sp.sparse.csr_matrix, b:np.ndarray, x:np.ndarray, _:any=None) -> tuple[np.array, np.array, any]:
        r = b - A * x
        y = A * r

        a = r @ r           
        c = r @ y           
        x = x + a / c * r

        return x, r, _