import numpy as np
import scipy as sp
import time

def toUsable(A):
    # CSR = compressed sparse Row format
    # CSC = compressed sparse Column format

    Acsr = A.tocsr(True)
    if(A.shape != Acsr.shape):
        Acsr = A.tocsc
        if(A.shape != Acsr.shape):
            print("Errore: dimensioni diverse nella trasposizione")

    return Acsr

def forward_substitution(L, b):
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

def diagonalize_sparse_matrix(matrix):
  # Ottenere la forma della matrice
  diagonal = matrix.copy()

  for i in range(diagonal.shape[0]):
      diagonal[i, i] = 1/diagonal[i, i]

  return diagonal

def jacobi(A, b, x0, tol, max_iter):
    """
    Implementa il metodo di Jacobi per la risoluzione di sistemi lineari su matrici sparse.

    Parametri:
        A (scipy.sparse.matrix): La matrice del sistema lineare.
        b (numpy.ndarray): Il vettore dei termini noti.
        x0 (numpy.ndarray): La soluzione iniziale.
        tol (float): La tolleranza per la convergenza.
        max_iter (int): Il numero massimo di iterazioni.

    Restituisce:
        numpy.ndarray: La soluzione approssimativa del sistema lineare.
    """

    # Dimensione del sistema
    n = A.shape[0]
    x = x0.copy()
    P = diagonalize_sparse_matrix(A)

    for it in range(max_iter):

        # Aggiornamento X
        for i in range(n):
            # Prendo il valore della diagonale
            d = A[i, i]
            #r = b - (A*x)
            r = b[i] - sum(A[i, :i] * x[:i]) - sum(A[i, i+1:] * x[i+1:])
            x[i] = r / d

        # Controlla la convergenza
        if (np.linalg.norm(A*x - b))/np.linalg.norm(b) < tol:
            print(f"Convergenza a {it} iterazioni")
            break

    return x

def gauss_seidel(A, b, x0, tol, max_iter):
    """
    Risolve un sistema lineare Ax = b usando il metodo di Gauss-Seidel.

    Parametri:
        A: matrice dei coefficienti (matrice sparsa).
        b: vettore dei termini noti.
        x0: vettore di partenza (soluzione iniziale).
        tol: tolleranza di convergenza.
        max_iter: numero massimo di iterazioni.

    Ritorna:
        x: vettore approssimato della soluzione.
    """

    # Dimensione del sistema
    x = x0.copy()

    P = sp.sparse.tril(A).tocsr()
    y = np.zeros(P.shape[0])  

    # Ciclo di iterazioni
    for it in range(max_iter):
        r = b - (A*x)

        # Sostituzione in avanti Py = r
        y = forward_substitution(L=P, b=r)
        x = x + y

        # Controllo della convergenza
        if  (np.linalg.norm(r))/np.linalg.norm(b) < tol:
            print(f"Convergenza a {it} iterazioni")
            break

    return x, r

def metodo_gradiente(A, b, x0, tol, max_iter):
    """
    Risolve un sistema lineare Ax = b usando il metodo del Gradiente.

    Parametri:
        A: matrice dei coefficienti (matrice sparsa).
        b: vettore dei termini noti.
        x0: vettore di partenza (soluzione iniziale).
        alpha: coefficiente di passo.
        tol: tolleranza di convergenza.
        max_iter: numero massimo di iterazioni.

    Ritorna:
        x: vettore approssimato della soluzione.
    """

    # Dimensione del sistema
    x = x0.copy() 

    for it in range(max_iter):
        # Calcolo residuo
        r = b - (A*x)
        y = A*r

        rt = r.T
        a = rt @ r
        c = rt @ y
        x = x + ((a/c)*r)

        # Controllo della convergenza
        if (np.linalg.norm(r))/np.linalg.norm(b) < tol:
            print(f"Convergenza a {it} iterazioni")
            break

    return x

def metodo_gradiente_coniugato(A, b, x0, tol, max_iter):
    """
    Risolve un sistema lineare simmetrico e definito positivo Ax = b usando il metodo del Gradiente coniugato.

    Parametri:
        A: matrice dei coefficienti (matrice sparsa simmetrica e definita positiva).
        b: vettore dei termini noti.
        x0: vettore di partenza (soluzione iniziale).
        tol: tolleranza di convergenza.
        max_iter: numero massimo di iterazioni.

    Ritorna:
        x: vettore approssimato della soluzione.
    """

    # Dimensione del sistema
    x = x0.copy()
    r = b - (A*x)
    d = r

    for it in range(max_iter):
        # Determino prossima soluzione
        r = b - (A*x)

        # Se non ho residuo sono già alla convergenza
        if r.all() == 0:
            print(f"Convergenza a {it} iterazioni")
            break

        y = A*d
        alpha = (d*r)/(d*y)
        x = x + (alpha*d)

        # Controllo della convergenza
        if (sp.linalg.norm(r))/sp.linalg.norm(b) < tol:
            print(f"Convergenza a {it} iterazioni")
            break

        # Determino direzione prossimo passo
        r = b - (A*x)
        w = A*r
        beta = (d*w)/(d*y)
        d = r - (beta*d)        

    return x

if __name__ == '__main__':
    # Caricamento in memoria
    A = sp.io.mmread('./matrici/spa1.mtx')

    # Dopo il caricamento cambio il formato per sfruttare l'aritmetica veloce
    A = toUsable(A)

    # Creo il vettore b da x con tutti 1
    x_test = np.ones(A.shape[0], dtype=np.float64)
    b = A.dot(x_test)

    x0 = np.zeros(A.shape[0], dtype=np.float64)

    print("Jacobi:")
    start = time.time()
    x = jacobi(A, b, x0, 10e-4, 20000)
    end = time.time()
    print(f"Tempo: {end-start} secondi")
    print()

    print("Gauss-Seidel:")
    start = time.time()
    x, r = gauss_seidel(A, b, x0, 10e-4, 20000)
    end = time.time()
    print(f"Tempo: {end-start} secondi")
    #print(f"Residuo: {r}")
    print()

    print("Gradiente:")
    start = time.time()
    x = metodo_gradiente(A, b, x0, 10e-4, 20000)
    end = time.time()
    print(f"Tempo: {end-start} secondi")
    print()

    print("Gradiente coniugato:")
    start = time.time()
    x = metodo_gradiente_coniugato(A, b, x0, 10e-4, 20000)
    end = time.time()
    print(f"Tempo: {end-start} secondi")
    print()