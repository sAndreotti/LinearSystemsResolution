from solver.JacobiSolver import JacobiSolver
from solver.GaussSeidelSolver import GaussSeidelSolver
from solver.GradientSolver import GradientSolver
from solver.ConjugateGradientSolver import ConjugateGradientSolver

import solver.Util as Util

import scipy as sp

if __name__ == '__main__':
    
    # Carico A     
    spa1 = sp.io.mmread('./matrici/spa1.mtx').tocsr()
    spa2 = sp.io.mmread('./matrici/spa2.mtx').tocsr()

    vem1 = sp.io.mmread('./matrici/vem1.mtx').tocsr()
    vem2 = sp.io.mmread('./matrici/vem2.mtx').tocsr()

    As = [spa1, spa2, vem1, vem2]

    tols = [1e-3, 1e-6, 1e-9, 1e-12]
    max_iter = 20000

    for i in range(len(As)):
        A = As[i]
        b = Util.create_b(A=A)

        for tol in tols:
            print(f"Matrice {i+1}")
            print(f"Tolerance: {tol}, max_iter: {max_iter}")
            jacobi = JacobiSolver(tol=tol, max_iter=max_iter)
            x = jacobi.solve(A=A, b=b)
            print(f'Jacobi:')
            print(f'Iterations: {jacobi.iter}')
            print(f'x: {x}')

            gauss_seidel = GaussSeidelSolver(tol=tol, max_iter=max_iter)
            x = gauss_seidel.solve(A=A, b=b)
            print(f'Gauss-Seidel:')
            print(f'Iterations: {gauss_seidel.iter}')
            print(f'x: {x}')

            gradient = GradientSolver(tol=tol, max_iter=max_iter)
            gradient.solve(A=A, b=b)
            print(f'Gradient:')
            print(f'Iterations: {gradient.iter}')
            print(f'x: {x}')

            conjugate = ConjugateGradientSolver(tol=tol, max_iter=max_iter)
            conjugate.solve(A=A, b=b)
            print(f'Conjugate Gradient:')
            print(f'Iterations: {conjugate.iter}')
            print(f'x: {x}')
            
            print()