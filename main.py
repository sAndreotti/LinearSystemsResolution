import structure as stru
import scipy as sp
import numpy as np

if __name__ == '__main__':
    # Creo A e b
    A = sp.io.mmread('./matrici/spa1.mtx').tocsr()
    b = A.dot(np.ones(A.shape[0], dtype=np.float64))

    tols = [10e-4, 10e-6, 10e-8, 10e-10]

    for tol in tols:
        print(f"Tolleranza {tol}")
        jacobi = stru.JacobiSolver(tol=tol, max_iter=20000)
        jacobi.solve(A=A, b=b)

        gauss = stru.GaussSeiderSolver(tol=tol, max_iter=20000)
        gauss.solve(A=A, b=b)

        gradient = stru.GradientSolver(tol=tol, max_iter=20000)
        gradient.solve(A=A, b=b)

        conjugate = stru.GradientConjugateSolver(tol=tol, max_iter=20000)
        conjugate.solve(A=A, b=b)
        print()