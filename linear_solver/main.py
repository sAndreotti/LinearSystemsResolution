from solver.JacobiSolver import JacobiSolver
from solver.GaussSeidelSolver import GaussSeidelSolver
from solver.GradientSolver import GradientSolver
from solver.ConjugateGradientSolver import ConjugateGradientSolver
import solver.Util as Util
import exception.MaxIterationException as MaxIterationException

import sys
import scipy as sp
import time 

SPA1_PATH = '../matrici/spa1.mtx'
SPA2_PATH = '../matrici/spa2.mtx'
VEM1_PATH = '../matrici/vem1.mtx'
VEM2_PATH = '../matrici/vem2.mtx'

SHOW_X = False

TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
MAX_ITER = 20000

METHOD_NAME = ['Jacobi', 'Gauss-Seidel', 'Gradient', 'Conjugate Gradient']

def test(path:str=None) -> None:
    As = []
    As_names = []
    times = [[], [], [], []]
    iterations = [[], [], [], []]

    paths = [SPA1_PATH, SPA2_PATH, VEM1_PATH, VEM2_PATH] if path is None else [path]
    
    for path in paths:
        try:
            A = sp.io.mmread(path).tocsr()
        except FileNotFoundError:
            print(f"File \'{path}\' not found")
            continue
        As.append(A)

        r_slash_index = path.rfind('/')
        if r_slash_index == -1:
            r_slash_index = path.rfind('\\')
        As_names.append(path[r_slash_index + 1 : path.rfind('.')])

    jacobi = JacobiSolver(tol=1e-8, max_iter=MAX_ITER)
    gauss_seidel = GaussSeidelSolver(tol=1e-8, max_iter=MAX_ITER)
    gradient = GradientSolver(tol=1e-8, max_iter=MAX_ITER)
    conjugate = ConjugateGradientSolver(tol=1e-8, max_iter=MAX_ITER)

    for i in range(len(As)):
        A = As[i]
        b = Util.create_b(A=A)

        for metod_time in times:
            metod_time.append([])
        for metod_iter in iterations:
            metod_iter.append([])

        print(f"Computing matrix {i + 1}/{len(As)}: {As_names[i]}")
        
        for tol in TOLS:   
            print(f"\tTolerance: {tol}, max_iter: {MAX_ITER}")
            
            jacobi.tol = tol
            start_time = time.time()
            print(f'\t\tJacobi:')
            try:
                x = jacobi.solve(A=A, b=b)
            except MaxIterationException:
                print(f"\t\t\tMax iteration reached: {jacobi}")
            end_time = time.time()
            print(f'\t\t\tTime: {end_time - start_time} s')
            print(f'\t\t\tIterations: {jacobi.iter}')
            if SHOW_X:
                print(f'x: {x}')
            times[0][i].append(end_time - start_time)
            iterations[0][i].append(jacobi.iter)

            gauss_seidel.tol = tol
            start_time = time.time()
            print(f'\t\tGauss-Seidel:')
            try:
                x = gauss_seidel.solve(A=A, b=b)
            
            except MaxIterationException:
                print(f"\t\t\tMax iteration reached: {gauss_seidel}")
            end_time = time.time()
            print(f'\t\t\tTime: {end_time - start_time} s')
            print(f'\t\t\tIterations: {gauss_seidel.iter}')
            if SHOW_X:
                print(f'x: {x}')
            times[1][i].append(end_time - start_time)
            iterations[1][i].append(gauss_seidel.iter)

            gradient.tol = tol
            start_time = time.time()
            print(f'\t\tGradient:')
            try:
                gradient.solve(A=A, b=b)
            except MaxIterationException:
                print(f"\t\t\tMax iteration reached: {gradient}")
            end_time = time.time()
            
            print(f'\t\t\tTime: {end_time - start_time} s')
            print(f'\t\t\tIterations: {gradient.iter}')
            if SHOW_X:
                print(f'x: {x}')
            times[2][i].append(end_time - start_time)
            iterations[2][i].append(gradient.iter)

            conjugate.tol = tol
            start_time = time.time()
            print(f'\t\tConjugate Gradient:')
            try:
                conjugate.solve(A=A, b=b)
            except MaxIterationException:
                print(f"\t\t\tMax iteration reached: {conjugate}")
            end_time = time.time()
            print(f'\t\t\tTime: {end_time - start_time} s')
            print(f'\t\t\tIterations: {conjugate.iter}')
            if SHOW_X:
                print(f'x: {x}')
            times[3][i].append(end_time - start_time)
            iterations[3][i].append(conjugate.iter)
            
            print()

    print("Resume:")
    for i in range(4):
        print(f"\tMethod: {METHOD_NAME[i]}")
        for j in range(len(As)):
            print(f"\t\t{As_names[j]}: {sum(times[i][j]) / len(times[i][j])} s, {sum(iterations[i][j]) / len(iterations[i][j])} iterations")

if __name__ == '__main__':    
    args = sys.argv[1:]

    if len(args) > 0:
        arg = args[0].lower()
        if arg == 'test' or arg == 't':
            test() 
        else:
            test(arg)
    else:
        path = input('Enter matrix path: ')
        test(path)
    