---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

# Homework
## Problems [Part 1](./01_Linear-Algebra.md)

1. Consider the following pistons connected in series. You want to find
   the distance each piston moves when equal and opposite forces are
   applied to each side. In case 1, you consider the motion of all five
   pistons, but in case 2 you set the motion of piston 5 to 0. 

![Springs-masses](../images/series-springs_01-02.png)

Using a FBD for each piston, in case 1:

$k_1x_{1}-k_2x_{2}=F$

$(k_1 +k_2)x_2 -k_1x_1 - k_2x_3 = 0$

$(k_2 +k_3)x_3 -k_2x_2 - k_3x_4 = 0$

$(k_3 +k_4)x_4 -k_3x_3 - k_4x_5 = 0$

$k_4x_5 - k_4x_4 = -F$

in matrix form:

$\left[ \begin{array}{ccccc}
k_1 & -k_1 & 0 & 0 & 0\\
-k_1 & k_1+k_2 & -k_2 & 0 & 0 \\
0 & -k_2 & k_2+k_3 & -k_3 & 0\\
0 & 0 & -k_3 & k_3+k_4 & -k_4\\
0 & 0 & 0 & -k_4 & k_4  
\end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \\
x_{5}\end{array} \right]=
\left[ \begin{array}{c}
F \\
0 \\
0 \\
0 \\
-F \end{array} \right]$

Try to use `np.linalg.solve` to find the piston x-positions. Do you get
any warnings or errors?

```{code-cell} ipython3
!head ../images/series-springs_01-02.png
```

```{code-cell} ipython3
#1.1
k1 = k2 = k3 = k4 = 1
F = 1
A=np.array([[k1,-k1,0,0,0], [-k1,(k1+k2),-k2, 0, 0],[0,-k2,(k2+k3),-k3,0],[0,0,-k3,(k3+k4),-k4],[0,0,0,-k4,k4]])
b=np.array([F,0,0,0,-F])
x = np.linalg.solve(A,b)
```

There is a singular matrix error.

+++

Now, consider case 2, 

Using a FBD for each piston, in case 2:

$k_1x_{1}-k_2x_{2}=F$

$(k_1 +k_2)x_2 -k_1x_1 - k_2x_3 = 0$

$(k_2 +k_3)x_3 -k_2x_2 - k_3x_4 = 0$

$(k_3 +k_4)x_4 -k_3x_3 = 0$

in matrix form:

$\left[ \begin{array}{cccc}
k_1 & -k_1 & 0 & 0 \\
-k_1 & k_1+k_2 & -k_2 & 0 \\
0 & -k_2 & k_2+k_3 & -k_3 \\
0 & 0 & -k_3 & k_3+k_4 \\
\end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \end{array} \right]=
\left[ \begin{array}{c}
F \\
0 \\
0 \\
0 \end{array} \right]$

Try to use `np.linalg.solve` to find the piston x-positions. Do you get
any warnings or errors? Why does this solution work better [hint: check
condition
numbers](./01_Linear-Algebra.md#Singular-and-ill-conditioned-matrices)

```{code-cell} ipython3
import numpy as np
A=np.array([[k1,-k1,0,0],
            [-k1,(k1+k2),-k2, 0],
            [0,-k2,(k2+k3),-k3],
            [0,0,-k3,(k3+k4)]])
b=np.array([F,0,0,0])

x = np.linalg.solve(A,b)
print('x1 = {:.2f} mg/m^3,\nx2 = {:.2f} mg/m^3,\nx3 = {:.2f} mg/mm^3'.format(*x))
```

This solution is better because in case one there was an infinite number of values for x and y. In case 2 we added a small number to make piston 5 zero, eliminating the fifth equation. This made it so the values of x and y can be solved for and there is now only one possibility of what they can be.

+++

![HVAC diagram showing the flow rates and connections between floors](../images/hvac.png)

2. In the figure above you have an idealized Heating, Ventilation and Air conditioning (HVAC) system. In the current configuration, the three-room building is being cooled off by $15^oC$ air fed into the building at 0.1 kg/s. Our goal is to determine the steady-state temperatures of the rooms given the following information

* $\dot{m}_1=0.1~kg/s$
* $\dot{m}_2=0.12~kg/s$
* $\dot{m}_3=0.12~kg/s$
* $\dot{m}_4=0.1~kg/s$
* $\dot{m}_5=0.02~kg/s$
* $\dot{m}_6=0.02~kg/s$
* $C_p=1000~\frac{J}{kg-K}$
* $\dot{Q}_{in} = 300~W$
* $T_{in} = 12^{o} C$

The energy-balance equations for rooms 1-3 create three equations:

1. $\dot{m}_1 C_p T_{in}+\dot{Q}_{in}-\dot{m}_2 C_p T_{1}+\dot{m}_6 C_p T_{2} = 0$

2. $\dot{m}_2 C_p T_{1}+\dot{Q}_{in}+\dot{m}_5 C_p T_{3}-\dot{m}_3 C_p T_{2}-\dot{m}_6 C_p T_{2} = 0$

3. $\dot{m}_3 C_p T_{2}+\dot{Q}_{in}-\dot{m}_5 C_p T_{3}-\dot{m}_4 C_p T_{3} = 0$

Identify the unknown variables and constants to create a linear algebra problem in the form of $\mathbf{Ax}=\mathbf{b}$.

a. Create the matrix $\mathbf{A}$

b. Create the known vector $\mathbf{b}$

c. Solve for the unknown variables, $\mathbf{x}$

d. What are the warmest and coldest rooms? What are their temperatures?

```{code-cell} ipython3
#1.2a
m1 = m4 = 0.1
m2 = m3 = 0.12
m5 = m6 = 0.02
C_p = 1000
Q_in = 300
T_in = 12
A=np.array([[-m2*C_p,m6*C_p, 0],
            [m2*C_p,(-m3*C_p) + (-m6*C_p), m5*C_p],
            [0,m3*C_p,(-m5*C_p)+(-m4*C_p)]])
```

```{code-cell} ipython3
#1.2b
b = ([(-m1*C_p*T_in) - Q_in,-Q_in,-Q_in])
```

```{code-cell} ipython3
#1.2c
x = np.linalg.solve(A,b)
for i in range(0,3):
    print('[{:5.1f} {:5.1f} {:5.1f}] {} [{:3.1f}] {} [{:5.1f}]'.format(*A[i],'*',x[i],'=',b[i]))
```

```{code-cell} ipython3
x = np.linalg.solve(A,b)
print('The unknown x values are:\nx1 = {:.2f} mg/m^3\nx2 = {:.2f} mg/m^3\nx3 = {:.2f} mg/mm^3'.format(*x))
```

```{code-cell} ipython3
#1.2d)
x1,x2,x3 = x
print('The warmest room is {:.2f}C and the coldest room is {:.2f}C.'.format(x3,x1))
```

3. The [Hilbert Matrix](https://en.wikipedia.org/wiki/Hilbert_matrix) has a high condition number and as the matrix increases dimensions, the condition number increases. Find the condition number of a 

a. $1 \times 1$ Hilbert matrix

b. $5 \times 5$ Hilbert matrix

c. $10 \times 10$ Hilbert matrix

d. $15 \times 15$ Hilbert matrix

e. $20 \times 20$ Hilbert matrix

If the accuracy of each matrix element is $\approx 10^{-16}$, what is the expected rounding error in the solution $\mathbf{Ax} = \mathbf{b}$, where $\mathbf{A}$ is the Hilbert matrix.

```{code-cell} ipython3
array_values = np.array([1,5,10,15,20])
def hilbert_matrix(N):
    H=np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            H[i,j]=1/(i+j+1)
    return H
```

```{code-cell} ipython3
for i in array_values:
    A = hilbert_matrix(i)
    x = np.zeros(i)
    b = A*x
    sol = np.linalg.solve(A,b)
    print("The condition numbers are {:.5} for {} x {}.".format(np.linalg.cond(A,2), i, i))
    print("The accuracy of each matrix is {:e}.\n".format(np.linalg.cond(A)))
```

## Problems [Part 2](./02_Gauss_elimination.md)

1. 4 masses are connected in series to 4 springs with K=100N/m. What are the final positions of the masses? 

![Springs-masses](../images/mass_springs.png)

The masses haves the following amounts, 1, 2, 3, and 4 kg for masses 1-4. Using a FBD for each mass:

$m_{1}g+k(x_{2}-x_{1})-kx_{1}=0$

$m_{2}g+k(x_{3}-x_{2})-k(x_{2}-x_{1})=0$

$m_{3}g+k(x_{4}-x_{3})-k(x_{3}-x_{2})=0$

$m_{4}g-k(x_{4}-x_{3})=0$

in matrix form K=100 N/m:

$\left[ \begin{array}{cccc}
2k & -k & 0 & 0 \\
-k & 2k & -k & 0 \\
0 & -k & 2k & -k \\
0 & 0 & -k & k \end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \end{array} \right]=
\left[ \begin{array}{c}
m_{1}g \\
m_{2}g \\
m_{3}g \\
m_{4}g \end{array} \right]$

```{code-cell} ipython3
#2.1
k = 100
K = np.array([[2*k, -k, 0, 0], 
              [-k, 2*k,-k, 0], 
              [0, -k, 2*k, -k],
              [0,0,-k,k]])
print(K)
```

```{code-cell} ipython3
b = np.array([1,2,3,4])*9.81
x = np.linalg.solve(K,b)
print('The final positions of the masses are:\nx1 = {:.2f} mg/m^3\nx2 = {:.2f} mg/m^3\nx3 = {:.2f} mg/mm^3'.format(*x))
```

![Triangular truss](../images/truss.png)

For problems __2-3__, consider the simple 3-element triangular truss, shown above, with a point load applied at the tip. The goal is to understand what tension is in the horizontal element, $P_1$. In problem __2__, the applied force is verical $(\theta=0)$ and in problem __3__ the applied force varies in angle $(\theta \neq 0)$. 

2. In the truss shown above, calculate the tension in bar 1, $P_1$, when $\theta=0$. When $\theta=0$, the $\sum F=0$ at each corner creates 3 equations and 3 unknowns as such (here, you reduce the number of equations with symmetry, $P_2=P_3,~R_2=R_3,~and~R_1=0$ ). 

$\left[ \begin{array}{ccc}
1 & \cos\alpha & 0 \\
0 & 2\sin\alpha & 0 \\
0 & -\sin\alpha &  1 \\
 \end{array} \right]
\left[ \begin{array}{c}
P_{1} \\
P_{2} \\
R_{2} \end{array} \right]=
\left[ \begin{array}{c}
0 \\
F \\
0 \end{array} \right]$

a. Create the system of equations, $\mathbf{Ax}=\mathbf{b}$, when $\alpha=75^o$, $\beta=30^o$, and $F=1~kN$. Use __Gauss elimination__ to solve for $P_1,~P_2,~and~R_2$. What is the resulting augmented matrix, $\mathbf{A|y}$ after Gauss elimination?

b. Solve for the $\mathbf{LU}$ decomposition of $\mathbf{A}$. 

c. Use the $\mathbf{LU}$ solution to solve for the tension in bar 1 $(P_1)$ every 10 N values of force, F, between 100 N and 1100 N. Plot $P_1~vs~F$.

```{code-cell} ipython3
#2.2a)
𝛼 = 75*np.pi/180 #Radians
𝛽 = 30*np.pi/180 #Radians 
F = 100 #Newtons
A = np.array([[1, np.cos(𝛼), 0], [0, 2*np.sin(𝛼), 0], [0, -np.sin(𝛼), 1]])
b = np.array([0, F, 0])

x, Aug = GaussNaive(A, b)
print('P1 is {:.2f} mg/m^3\nP2 is {:.2f} mg/m^3\nR2 is {:.2f} mg/mm^3'.format(*x))

print('The resulting augmented matrix, 𝐀|𝐲 after Gauss elimination is\n', Aug)
```

```{code-cell} ipython3
def GaussNaive(A,y):
    '''GaussNaive: naive Gauss elimination
    x = GaussNaive(A,b): Gauss elimination without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    y = right hand side vector
    returns:
    ---------
    x = solution vector
    Aug = augmented matrix (used for back substitution)'''
    [m,n] = np.shape(A)
    Aug = np.block([A,y.reshape(n,1)])
    Aug = Aug.astype(float)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination 
    for k in range(0,n-1):
        for i in range(k+1,n):
            if Aug[i,k] != 0.0:
                factor = Aug[i,k]/Aug[k,k]
                Aug[i,:] = Aug[i,:] - factor*Aug[k,:]
    # Back substitution
    x=np.zeros(n)
    for k in range(n-1,-1,-1):
        x[k] = (Aug[k,-1] - Aug[k,k+1:n]@x[k+1:n])/Aug[k,k]
    return x,Aug
```

```{code-cell} ipython3
#2.2b)
L,U = LUNaive(A)

print('L decomposition is:\n',L)
print('U decomposition is:\n',U)
```

```{code-cell} ipython3
print(A)
print(L@U)
```

```{code-cell} ipython3
def LUNaive(A):
    '''LUNaive: naive LU decomposition
    L,U = LUNaive(A): LU decomposition without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    returns:
    ---------
    L = Lower triangular matrix
    U = Upper triangular matrix
    '''
    [m,n] = np.shape(A)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination
    U = A.astype(float)
    L = np.eye(n)

    for k in range(0,n-1):
        for i in range(k+1,n):
            if U[k,k] != 0.0:
                factor = U[i,k]/U[k,k]
                L[i,k]=factor
                U[i,:] = U[i,:] - factor*U[k,:]
    return L,U
```

```{code-cell} ipython3
#2.2c)
import matplotlib
from numpy.random import default_rng
rng = default_rng()
```

```{code-cell} ipython3
Fspace = np.zeros(1000)
xspace = np.zeros(1000)
for i in range(1000):
    F = 1000 * rng.random(3)
    y = np.linalg.solve(L, F)
    x = np.linalg.solve(U, y)
    Fspace[i] = F[1]
    xspace[i] = x[0]
```

```{code-cell} ipython3
plt.plot(xspace, Fspace, 's')
plt.xlabel('P1(N)')
plt.ylabel('F(N)')
plt.title('P1(N) vs F(N)')
```

3. Using the same truss as shown above, let's calculate the tension in bar 1, $P_1$, when $\theta=[0...90^o]$ and $F=[100...1100]~kN$. When $\theta\neq 0$, the resulting 6 equations and 6 unknowns are given in the following matrix

$\left[ \begin{array}{ccc}
0 & \sin\alpha & 0 & 1 & 0 & 0 \\
1 & \cos\alpha & 0 & 0 & 1 & 0 \\
0 & \cos\beta/2 & \cos\beta/2 & 0 & 0 & 0 \\
0 & -\sin\beta/2 & \sin\beta/2 & 0 & 0 & 0 \\
-1 & 0 & \cos\alpha & 0 & 0 & 0 \\
0 & 0 & \sin\alpha & 0 & 0 & 1 \\
 \end{array} \right]
\left[ \begin{array}{c}
P_{1} \\
P_{2} \\
P_{3} \\
R_{1} \\
R_{2} \\
R_{3}\end{array} \right]=
\left[ \begin{array}{c}
0 \\
0 \\
F\cos\theta \\
-F\sin\theta \\
0 \\
0 \end{array} \right]$

a. Create the system of equations, $\mathbf{Ax}=\mathbf{b}$, when $\alpha=75^o$, $\beta=30^o$, $\theta=45^o=\pi/4$, and $F=1~kN$. Use __Gauss elimination__ to solve for $P_1,~P_2,~P_3,~R_1,~R_2,~and~R_3$. What is the resulting augmented matrix, $\mathbf{A|y}$ after Gauss elimination? _Hint: do you see a problem with the order of the equations? How can you __pivot__ to fix it?_

b. Solve for the $\mathbf{PLU}$ decomposition of $\mathbf{A}$. 

c. Use the $\mathbf{PLU}$ solution to solve for the tension in bar 1 $(P_1)$ every 10 N values of force, F, between 100 N and 1100 N. Plot $P_1~vs~F$.

```{code-cell} ipython3
#2.3a)
𝛼 = 75*np.pi/180 #Radians
𝛽 = 30*np.pi/180 #Radians
𝜃 = 45*np.pi/180
F = 100 #Newtons
A = np.array([[1, np.cos(𝛼), 0, 0, 1, 0], #pivoted by switching the first two rows
              [0, np.sin(𝛼), 0, 1, 0, 0], 
              [0, np.cos(𝛽/2), np.cos(𝛽/2), 0, 0, 0], 
              [0, -np.sin(𝛽/2), np.sin(𝛽/2), 0, 0, 0], 
              [-1, 0, np.cos(𝛼), 0, 0, 0], 
              [0, 0, np.sin(𝛼), 0, 0, 1]])
b = np.array([0, 0, F*np.cos(𝜃), -F*np.sin(𝜃), 0, 0])
x, Aug = GaussNaive(A, b)
print('P1 is {:.2f} mg/m^3\nP2 is {:.2f} mg/m^3\nR2 is {:.2f} mg/mm^3'.format(*x))

print('The resulting augmented matrix, 𝐀|𝐲 after Gauss elimination is\n', Aug)
```

```{code-cell} ipython3
#2.3b)
from scipy.linalg import lu

P,L,U = lu(A) # a built-in partial-pivoting LU decomposition function
print('P=\n',P)
print('L=\n',L)
print('U=\n',U)
```

```{code-cell} ipython3
print(A)
print(P@L@U)
```

```{code-cell} ipython3
#2.3c)
Fspace = np.zeros(1000)
xspace = np.zeros(1000)
for i in range(1000):
    F = 1000 * rng.random(6)
    y = np.linalg.solve(L,F)
    x = np.linalg.solve(U, y)
    Fspace[i] = F[1]
    xspace[i] = x[0]
```

```{code-cell} ipython3
plt.plot(xspace, Fspace, 's')
plt.xlabel('P1(N)')
plt.ylabel('F(N)')
plt.title('P1(N) vs F(N)')
```

## Problems [Part 3](./03_Linear-regression-algebra.md)

<img
src="https://i.imgur.com/LoBbHaM.png" alt="prony series diagram"
style="width: 300px;"/> <img src="https://i.imgur.com/8i140Zu.png" alt
= "stress relax data" style="width: 400px;"/> 

Viscoelastic Prony series model and stress-vs-time relaxation curve of wheat kernels [[3]](https://www.cerealsgrains.org/publications/plexus/cfw/pastissues/2013/Documents/CFW-58-3-0139.pdf). Stress relaxation curve of a wheat kernel from regressed equation data that illustrate where to locate relaxation times (vertical dotted lines) and stresses (horizontal black marks). $\sigma$ = stress; t = time.

2. [Viscoelasticity](https://en.wikipedia.org/wiki/Viscoelasticity) is a property of materials that exhibit stiffness, but also tend to flow slowly. One example is [Silly Putty](https://en.wikipedia.org/wiki/Silly_Putty), when you throw a lump it bounces, but if you leave it on a table it _creeps_, slowly flowing downwards. In the stress-vs-time plot above, a wheat kernel was placed under constant strain and the stress was recorded. In a purely elastic material, the stress would be constant. In a purely viscous material, the stress would decay to 0 MPa. 

Here, you have a viscoelastic material, so there is some residual elastic stress as $t\rightarrow \infty$. The researchers used a 4-part [Prony series](https://en.wikipedia.org/wiki/Prony%27s_method) to model viscoelasticity. The function they fit was

$\sigma(t) = a_1 e^{-t/1.78}+a_2 e^{-t/11}+a_3e^{-t/53}+a_4e^{-t/411}+a_5$

a. Load the data from the graph shown above in the file `../data/stress_relax.dat`. 

b. Create a $\mathbf{Z}$-matrix to perform the least-squares regression for the given Prony series equation $\mathbf{y} = \mathbf{Za}$.

c. Solve for the constants, $a_1,~a_2,~a_3,~a_4~,a_5$

d. Plot the best-fit function and the data from `../data/stress_relax.dat` _Use at least 50 points in time to get a smooth best-fit line._

```{code-cell} ipython3
#3.2a)
!head ../data/stress_relax.dat
data_file = np.loadtxt('../data/stress_relax.dat', skiprows = 1, delimiter = ',')
```

```{code-cell} ipython3
stress = data_file[:,1]
time = data_file[:,0]
plt.plot(time, stress, 's')
```

```{code-cell} ipython3
#3.2b)
Z = np.block([[np.exp(-time/1.78)],
              [np.exp(-time/11)],
              [np.exp(-time/53)],
              [np.exp(-time/411)],
              [np.exp(time*0)]]).T
Z
```

```{code-cell} ipython3
from scipy.linalg import lstsq
from sympy import solve
```

```{code-cell} ipython3
#3.2c)
constants = np.linalg.solve(Z.T@Z, Z.T@stress)
print('The constants of the least-squares regression are:', constants)
```

```{code-cell} ipython3
lstsq(Z, stress) #least-squares solution
```

```{code-cell} ipython3
#3.2d)
plt.plot(time, stress, 's', label = 'Data in File')
plt.plot(time, Z@constants, label = 'Least-Squares Regression')
plt.xlabel('Time(s)')
plt.ylabel('Stress(MPa)')
plt.title('Data w/ Best Fit')
plt.legend()
```

3. Load the '../data/primary-energy-consumption-by-region.csv' that has the energy consumption of different regions of the world from 1965 until 2018 [Our world in Data](https://ourworldindata.org/energy). 
You are going to compare the energy consumption of the United States to all of Europe. Load the data into a pandas dataframe. *Note: you can get certain rows of the data frame by specifying what you're looking for e.g. 
`EUR = dataframe[dataframe['Entity']=='Europe']` will give us all the rows from Europe's energy consumption.*

a. Use a piecewise least-squares regression to find a function for the energy consumption as a function of year

energy consumed = $f(t) = At+B+C(t-1970)H(t-1970)$

c. What is your prediction for US energy use in 2025? How about European energy use in 2025?

```{code-cell} ipython3
#3.3
!head '../data/primary-energy-consumption-by-region.csv' 
data_file = pd.read_csv('../data/primary-energy-consumption-by-region.csv')
```

```{code-cell} ipython3
#3.3a)
US = data_file[data_file['Entity']=='United States']
US_year = US['Year'].values
US_energy = US['Primary Energy Consumption (terawatt-hours)'].values
```

```{code-cell} ipython3
Z = np.block([[US_year],[US_year**0], [(US_year>=1970)*(US_year-1970)]]).T
```

```{code-cell} ipython3
plt.plot(US_year, US_energy, 's')
constants = np.linalg.solve(Z.T@Z, Z.T@US_energy)
plt.plot(US_year, Z@constants)
```

```{code-cell} ipython3
Europe = data_file[data_file['Entity']=='Europe']
Europe_year = Europe['Year'].values
Europe_energy = Europe['Primary Energy Consumption (terawatt-hours)'].values
```

```{code-cell} ipython3
Z_Europe = np.block([[Europe_year],[Europe_year**0], [(Europe_year>=1970)*(Europe_year-1970)]]).T
```

```{code-cell} ipython3
plt.plot(Europe_year, Europe_energy, 's')
constants_europe = np.linalg.solve(Z_Europe.T@Z, Z_Europe.T@Europe_energy)
plt.plot(Europe_year, Z_Europe@constants_europe)
```

```{code-cell} ipython3
#3.3b)
plt.plot(US_year, US_energy, 's')
constants = np.linalg.solve(Z.T@Z, Z.T@US_energy)
extension = np.arange(1960, 2030)
Z_extension= np.block([[extension],[extension**0], [(extension>=1970)*(extension-1970)]]).T
plt.axvline(x = 2025, color = 'b', label = 'axvline - full height')
plt.plot(extension, Z_extension@constants)
```

My prediciton for US energy use in 2025 is about 28000 TW.

```{code-cell} ipython3
plt.plot(Europe_year, Europe_energy, 's')
constants = np.linalg.solve(Z_Europe.T@Z, Z_Europe.T@US_energy)
extension_europe = np.arange(1960, 2030)
Z_extension_europe= np.block([[extension_europe],[extension_europe**0], [(extension_europe>=1970)*(extension_europe-1970)]]).T
plt.axvline(x = 2025, color = 'b', label = 'axvline - full height')
plt.plot(extension_europe, Z_extension_europe@constants)
```

My prediciton for Europe energy use in 2025 is about 29000 TW.
