# Linear systems resolution
Implementation of iterative methods for solve linear systems

## Stationary iterative methods
Stationary methods are based on splitting, given a matrix $A \in \mathbb{R}^{nxn}$ they suppose to split the matrix in $A = P - N$ where $P,N \in \mathbb{R}^{nxn}$ makeing the equation $Ax=b$ equal to $x=P^{-1}Nx+p^{-1}b$. 
$P$ is the matrix that have the diagonal of $A$ and all other entry equals to 0. $N$ is the matrix where all the entry on the main diagonal are equal to 0, and all the other entry are equal to the opposite of $A$.
They base the next $x$ value on the equation $x^{(k+1)}=x^{(k)}+ \alpha P^{-1}r^{(k)}$

### Jacobi method
Given a matrix $A \in \mathbb{R}^{nxn}$ with strict diagonal dominance for rows the Jacobi method converges

### Gauss-Seidel method
Given a matrix $A \in \mathbb{R}^{nxn}$ with strict diagonal dominance for rows the Gauss-Seidel method converges

## Not stationary iterative methods
Stationary methods have a fixed $\alpha$, the next 2 methods calculate the appropriate $\alpha$ every iteration k.
They base the next $x$ value on the equation $x^{(k+1)}=x^{(k)}+ \alpha_k P^{-1}r^{(k)}$

## Gradient method
Given a matrix $A \in \mathbb{R}^{nxn}$ simmetrical and positive defined, the gradient method converges for any initial guess $x^{(0)}$

## Conjunate gradient method
Given a matrix $A \in \mathbb{R}^{nxn}$ simmetrical and positive defined, the conjunate gradient method converges at most in $n$ iteractions

## How to use
<code>structure.py</code> is the main script that contains the library, it implements the methods described before. It is built around the Solver class which define a geenric iterative solver, 
then the 4 specific Solver class are defined base on the structure:
- Jacobi method: **JacobiSolver**
- Gauss-Seider method: **GaussSeiderSolver**
- Gradient method: **GradientSolver**
- Conjugate Gradient method: **GradientConjugateSolver**
  
They all require only a **$A$** matrix with diagonals not equals to 0 and matching the algorithm requisites and a **$b$** vector,
more over must be specified a **tollerance** to stop the method and a **max_iteration** to do before stopping.
**tollerance** and **max_iteration** are defined with the initialization as shown in <code>main.py</code>

<code>main.py</code> contains an example on how to use the library <code>structure.py</code>. 
This example run all the 4 methods with the matrix <code>spa1.mtx</code> and different tollerances

## More examples
More clear examples on how are calculated the result and see how are distributed the values in the sparse matrix are shown in <code>progetto1bis.ipynb</code> and in <code>progetto1bis.py</code>

## Dependencies
List of all the **dependecies** to use the library (can be installed with <code>pip install</code> or <code>conda install</code>):
- [numpy](https://numpy.org/): Linear algebra for python;
- [scipy](https://scipy.org/): Sparse matrix algebra;

