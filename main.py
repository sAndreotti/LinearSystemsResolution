import structure as stru
import scipy as sp
import numpy as np

if __name__ == '__main__':
    A = sp.io.mmread('./matrici/spa1.mtx').tocsr()

    # Creo il vettore b da x_test con tutti 1
    x_test = np.ones(A.shape[0], dtype=np.float64)
    b = A.dot(x_test)

    jacobi = stru.JacobiSolver(tol=10e-4, max_iter=20000)
    jacobi.solve(A=A, b=b)

    gauss = stru.GaussSeiderSolver(tol=10e-4, max_iter=20000)
    gauss.solve(A=A, b=b)

    gradient = stru.GradientSolver(tol=10e-4, max_iter=20000)
    gradient.solve(A=A, b=b)

    conjugate = stru.GradientConjugateSolver(tol=10e-4, max_iter=20000)
    conjugate.solve(A=A, b=b)