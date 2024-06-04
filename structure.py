# Classe madre
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import time

class Solver(ABC):

    # Inizializzo A, b, tol e max_iter
    def __init__(self, tol, max_iter):
        self.tol = tol
        self.max_iter = max_iter


    @abstractmethod
    def initialize_method(self):
        pass


    @abstractmethod
    def check_termination(self, A, x, b):
        pass

    @abstractmethod
    def update_x(self, A, x):
        pass

    def solve(self, A, b):
        start = time.time()

        self.A = A
        self.b = b
        x = self.initialize_method()

        for it in range(self.max_iter):
            x, residuo = self.update_x(x)

            if self.check_termination(residuo, b):
                print(f"Converge a {it} iterazioni")
                print(f"Tempo di esecuzione {time.time()-start} secondi")
                return x

        raise Exception("Non converge")

#--------------------------------------------------------------
class JacobiSolver(Solver): 

    def initialize_method(self):
        print("Metodo di Jacobi")
        return np.zeros(self.A.shape[0], dtype=np.float64)

    # Controlla la convergenza
    def check_termination(self, residuo, b):
        if np.linalg.norm(residuo)/np.linalg.norm(b) < self.tol:
            return True

    # Aggiornamento X
    def update_x(self, x):
        n = self.A.shape[0]
        for i in range(n):
            # Prendo il valore della diagonale
            d = self.A[i, i]
            r = self.b[i] - sum(self.A[i, :i] * x[:i]) - sum(self.A[i, i+1:] * x[i+1:])
            x[i] = r / d

        return x, self.A*x-self.b

#--------------------------------------------------------------
class GaussSeiderSolver(Solver): 

    def forward_substitution(self, L, b):
        n = L.shape[0]
        x = np.zeros(L.shape[0])

        if L[0, 0] == 0:
            return []
        
        x[0]=b[0]/L[0, 0]
        for i in range(1, n):
            if L[i, i] == 0:
                return []
            x[i] = (b[i] - (L[i, :]*x))/L[i, i]

        return x

    def initialize_method(self):
        print("Metodo di Gauss-Seider")
        self.P = sp.sparse.tril(self.A).tocsr()
        self.y = np.zeros(self.P.shape[0]) 
        return np.zeros(self.A.shape[0], dtype=np.float64)

    # Controlla la convergenza
    def check_termination(self, residuo, b):
        if np.linalg.norm(residuo)/np.linalg.norm(b) < self.tol:
            return True

    # Aggiornamento X
    def update_x(self, x):
        residuo = self.b - (self.A*x)
        
        # Sostituzione in avanti Py = r
        y = self.forward_substitution(L=self.P, b=residuo)
        x = x + y

        return x, residuo

#--------------------------------------------------------------
class GradientSolver(Solver): 

    def initialize_method(self):
        print("Metodo del Gradiente")
        return np.zeros(self.A.shape[0], dtype=np.float64)

    # Controlla la convergenza
    def check_termination(self, residuo, b):
        if np.linalg.norm(residuo)/np.linalg.norm(b) < self.tol:
            return True

    # Aggiornamento X
    def update_x(self, x):
        # Calcolo residuo
        residuo = self.b - (self.A*x)
        y = self.A*residuo

        rt = residuo.T
        a = rt @ residuo
        c = rt @ y
        x = x + ((a/c)*residuo)

        return x, residuo
    
#--------------------------------------------------------------
class GradientConjugateSolver(Solver): 

    def initialize_method(self):
        print("Metodo del Gradiente Coniugato")
        self.first = True
        return np.zeros(self.A.shape[0], dtype=np.float64)
    
    # Controlla la convergenza
    def check_termination(self, residuo, b):
        if np.linalg.norm(residuo)/np.linalg.norm(b) < self.tol:
            return True

    # Aggiornamento X
    def update_x(self, x):
        # Determino prossima soluzione
        if self.first:
            self.residuo = self.b - (self.A*x)
            self.d = self.residuo
            self.first = False

        y = self.A*self.d
        alpha = (self.d*self.residuo)/(self.d*y)
        x = x + (alpha*self.d)

        # Controllo della convergenza
        if self.check_termination(residuo=self.residuo, b=self.b):
            return x, self.residuo

        # Determino direzione prossimo passo
        self.residuo = self.b - (self.A*x)
        w = self.A*self.residuo
        beta = (self.d*w)/(self.d*y)
        self.d = self.residuo - (beta*self.d)

        return x, self.residuo