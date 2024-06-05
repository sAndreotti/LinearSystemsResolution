from abc import ABC, abstractmethod
import numpy as np
import scipy as sp

import solver.Util as Util 
from exception.MaxIterationException import MaxIterationException

class Solver(ABC):

    def __init__(self, tol:float=Util.DEFAULT_TOL, max_iter:int=Util.DEFAULT_MAX_ITER, 
                 initialization_mode:Util.InitializationMode=Util.DEFAULT_INITIALIZATION_MODE,
                 lower_bound:float=Util.DEFAULT_LOWER_BOUND, upper_bound:float=Util.DEFAULT_UPPER_BOUND):
        '''
        Constructor of the class Solver.

        Parameters:
        @tol: float -> Tolerance of the solver.
        @max_iter: int -> Maximum number of iterations.
        @initialization_mode: Util.InitializationMode -> Initialization mode of the solver.
        @lower_bound: float -> Lower bound of the initialization (if necessary).
        @upper_bound: float -> Upper bound of the initialization (if necessary).
        '''
        
        self.tol = tol
        self.max_iter = max_iter
        
        self.initialization_mode = initialization_mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.iter = 0
    
    def _initialize_x_0(self, N:int) -> np.ndarray:
        if self.initialization_mode == Util.InitializationMode.ZEROS:
            return np.zeros(N, dtype=np.float64)
        elif self.initialization_mode == Util.InitializationMode.RANDOM:
            np.random.uniform(self.lower_bound, self.upper_bound, N)

        raise Exception("Initialization mode not supported")

    def _check_termination(self, r:np.array, b:np.ndarray) -> bool:
        return np.linalg.norm(r) / np.linalg.norm(b) < self.tol

    @abstractmethod
    def _update_x(self, A, b, x, support:any=None) -> tuple[np.ndarray, np.ndarray, any]:
        pass
    
    def solve(self, A:sp.sparse.csr_matrix, b:np.ndarray, support:any=None) -> np.ndarray:
    
        x = self._initialize_x_0(A.shape[1])
        k = 0
        r = b - A.dot(x)

        while not self._check_termination(r, b):
            x, r, support = self._update_x(A, b, x, support)
            k += 1
            
            if k > self.max_iter:
                self.iter = k
                raise MaxIterationException(f"Max iteration reached: {k}", k)

        self.iter = k
        return x