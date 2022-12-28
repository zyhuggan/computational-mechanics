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

> __Content modified under Creative Commons Attribution license CC-BY
> 4.0, code under BSD 3-Clause License © 2020 R.C. Cooper__

```{code-cell} ipython3
import math
```

# Homework

## Problems [Part 1](./01_Interacting_with_Python.md)

1. Calculate some properties of a rectangular box that is 12.5"$\times$11"$\times$14" and weighs 31 lbs

    a. What is the volume of the box?
    
    b. What is the average density of the box?
    
    c. What is the result of the following logical operation, `volume>1000` (in inches^3)

```{code-cell} ipython3
#a). Volume of a Rectangular Box 12.5" x 11" x 14"
length = 12.5 #inches 
width = 11 #inches
height = 14 #inches
volume = length * width * height #in^3

#b). Average Density of the Rectangular Box
mass = 31 #lbs
density = mass/volume #lbs/in^3

print(volume)
print(density)

#c). Logical Operation
print(volume > 1000)
```

The volume of the rectangular box is 1925 lbs.
The density of the rectangular box is lb/in^3.
The logical operation will return True since 1925lbs is more than 1000lbs.

+++

2. Use the variables given below, `str1` and `str2`, and check the following 

    a. `str1<str2`
    
    b. `str1==str2`
    
    c. `str1>str2`
    
    d. How could you force (b) to be true? [Hint](https://docs.python.org/3/library/stdtypes.html?highlight=str.lower#str.lower) or [Hint](https://docs.python.org/3/library/stdtypes.html?highlight=str.lower#str.upper)

```{code-cell} ipython3
str1 = 'Python'
str2 = 'python'
#a). 
print(str1 < str2)
#b).
print(str1 == str2)
#c). 
print(str1 > str2)
#d). 
str1 = str1.lower()
print(str1 == str2)
```

String one is less than string two and string one is not more than string two. They are not not equal to each other since the P in string one is uppercase but we can make them equal to each other by using the str.lower() command which would make the p lowercase. Using this command, the strings are now equal.

+++

3. The following code has an error, fix the error so that the correct result is returned:

```y is 20 and x is less than y```

```python
x="1"
y=20

if x<y and y==20:
    print('y is 20 and x is less than y')
else:
    print('x is not less than y')
```

```{code-cell} ipython3
#y is 20 and x is less than y

x = 1
y = 20

if x < y and y == 20:
    print('y is 20 and x is less than y')
else:
    print('x is not less than y')
```

The error in this code occurs because the x variable is a string and not an integer. A string type can not be compared to an integer type. In order to fix this error, we can make the x variable an integer.

+++

4. There is a commonly-used programming question that asks interviewees
   to build a [fizz-buzz](https://en.wikipedia.org/wiki/Fizz_buzz) result. 
   
   Here, you will build a similar program, but use the numbers from the
   class, **3255:** $3,~2,~5\rightarrow$ "computational", "mechanics",
   "rocks!". You should print out a list of numbers, if the number is
   divisible by 3, replace the 3 with "computational". If the number is
   divisible by 2, replace with "mechanics". If the number is divisible
   by 5, replace the number with "rocks!". If the number is divisible by
   a combination, then add both words e.g. 6 is divisible by 3 and 2, so
   you would print out "computational mechanics". 
   
   Here are the first 20 outputs your program should print, 
   
| index | printed output |
| ---   | ---            |
0 | Computational Mechanics Rocks!
1 | 1
2 | Mechanics 
3 | Computational 
4 | Mechanics 
5 | Rocks!
6 | Computational Mechanics
7 | 7
8 | Mechanics 
9 | Computational 
10 | Mechanics Rocks!
11 | 11
12 | Computational Mechanics
13 | 13
14 | Mechanics 
15 | Computational Rocks!
16 | Mechanics 
17 | 17
18 | Computational Mechanics
19 | 19

```{code-cell} ipython3
def fizz_buzz(number): #defined a functions
    '''Determines if a number is divisble by 3, 2, and/or 5 and returns the correct statement'''
    if (number % 3 == 0 ) and (number % 2 == 0) and (number % 5 == 0):
        return "Computational Mechanics Rocks!"
    elif (number % 3 == 0) and (number % 2== 0):
        return "Computatial Mechanics"
    elif (number % 3 == 0) and (number % 5 == 0):
        return "Computational Rocks!"
    elif (number % 2 == 0) and (number % 5 == 0):
        return "Mechanics Rocks!"
    elif (number % 3 == 0):
        return "Computational"
    elif (number % 2 == 0):
        return "Mechanics"
    elif (number % 5 == 0):
        return "Rocks!"
    else:
        return number

for i in range(20):
    print(fizz_buzz(i))
    
```

## Problems [Part 2](./02_Working_with_Python.md)

1. Create a function called `sincos(x)` that returns two arrays, `sinx` and `cosx` that return the sine and cosine of the input array, `x`. 

    a. Document your function with a help file in `'''help'''`
    
    b. Use your function to plot sin(x) and cos(x) for x=$0..2\pi$

```{code-cell} ipython3
import numpy as np 
import matplotlib.pyplot as plt
def sincos(x):
    '''Returns the sine and cosine arrays of an inputted number.
    a). Documents the function and b). plots the function from 0 to 2pi
    '''
    sinx = np.sin(x)
    cosx = np.cos(x)
    plt.plot(x, sinx, color='red', label='sin(x)')
    plt.plot(x, cosx, color='black', label='cos(X)')
    plt.legend()
    return(sinx, cosx)

x =np.linspace(0, 2*np.pi)
help(sincos)
sincos(x)
```

2. Use a for-loop to create a variable called `A_99`, where every element is the product
of the two indices from 0 to 9 e.g. A_99[3,2]=6 and A_99[4,4]=16. 

    a. time your script using `%%time`    
    
    b. Calculate the mean of `A_99`

    c. Calculate the standard deviation of `A_99`

```{code-cell} ipython3
%time
A_99=[]
for i in range(10):
    for j in range(10):
        A_99.append(i*j)

#mean calculation
summation = np.sum(A_99)
length = len(A_99)
mean = summation/length
print(mean)

#standard deviation calculation
square = []
for i in A_99:
    combine = (i - mean)**2
    square.append(combine)
total = np.sum(square)/length
standard_deviation = math.sqrt(total)
print(standard_deviation)
```

3. Use the two arrays, X and Y, given below to create A_99 using numpy array math rather than a for-loop.

```{code-cell} ipython3
X, Y = np.meshgrid(np.arange(10), np.arange(10))
```

    a. time your script using `%%time`    
    
    b. Calculate the mean of `A_99`

    c. Calculate the standard deviation of `A_99`
        
    d. create a filled contour plot of X, Y, A_99 [contourf plot documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html)

```{code-cell} ipython3
#a) Time script
%time
X, Y = np.meshgrid(np.arange(10), np.arange(10))
A_99 = X * Y

A_99_reshape = np.reshape(A_99, 100)

#b) Mean calculation
summation = np.sum(A_99_reshape)
length = len(A_99_reshape)
mean = summation/length
print(mean)

#c) Standard deviation calculation
square = []
for i in A_99:
    combine = (i - mean)**2
    square.append(combine)
total = np.sum(square)/length
standard_deviation = math.sqrt(total)
print(standard_deviation)

#d) Filled Contour Plot
Z =(X**2+Y**2)**0.5
plt.contourf(X,Y,Z)
```

4. The following linear interpolation function has an error. It is supposed to return y(x) given the the two points $p_1=[x_1,~y_1]$ and $p_2=[x_2,~y_2]$. Currently, it just returns and error.

```python
def linInterp(x,p1,p2):
    '''linear interplation function
    return y(x) given the two endpoints 
    p1=np.array([x1,y1])
    and
    p2=np.array([x2,y2])'''
    slope = (p2[2]-p1[2])/(p2[1]-p1[1])
    
    return p1[2]+slope*(x - p1[1])
```

```{code-cell} ipython3
#4) cCorrected interpolation function
def linInterp(x,p1,p2):
    '''linear interplation function
    return y(x) given the two endpoints 
    p1=np.array([x1,y1])
    and
    p2=np.array([x2,y2])'''
    slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    
    return p1[1]+slope*(x - p1[0])
print(linInterp(2, [3,4], [5,6]))
```

The problem with this interpolation function was that the indices indicated were out of range. By changing the indices to 1 and 0 the indicies are now in range and can be used in the function.

+++

## Problems [Part 3](03_Numerical_error.md)

1. The growth of populations of organisms has many engineering and scientific applications. One of the simplest
models assumes that the rate of change of the population p is proportional to the existing population at any time t:

$\frac{dp}{dt} = k_g p$

where $t$ is time in years, and $k_g$ is growth rate in \[1/years\]. 

The world population has been increasing dramatically, let's make a prediction based upon the [following data](https://worldpopulationhistory.org/map/2020/mercator/1/0/25/) saved in [world_population_1900-2020.csv](../data/world_population_1900-2020.csv):


|year| world population |
|---|---|
|1900|1,578,000,000|
|1950|2,526,000,000|
|2000|6,127,000,000|
|2020|7,795,482,000|

a. Use a growth rate of $k_g=0.013$ [1/years] and compare the analytical solution (use initial condition p(1900) = 1578000000) to the Euler integration for time steps of 20 years from 1900 to 2020 (Hint: use method (1)- plot the two solutions together with the given data) 

b. Discussion question: If you decrease the time steps further and the solution converges, will it converge to the actual world population? Why or why not? 

**Note: We have used a new function `np.loadtxt` here. Use the `help` or `?` to learn about what this function does and how the arguments can change the output. In the next module, we will go into more details on how to load data, plot data, and present trends.**

```{code-cell} ipython3
import numpy as np
year, pop = np.loadtxt('../data/world_population_1900-2020.csv',skiprows=1,delimiter=',',unpack=True)
print('years=',year)
print('population =', pop)
```

```{code-cell} ipython3
print('average population changes 1900-1950, 1950-2000, 2000-2020')
print((pop[1:] - pop[0:-1])/(year[1:] - year[0:-1]))
print('average growth of 1900 - 2020')
print(np.mean((pop[1:] - pop[0:-1])/(year[1:] - year[0:-1])))

#3a).Comparison of the Analytical Solution and the Euler Integration
time = np.linspace(1900, 2020, 20)
k_g = 0.013
analytical_solution =lambda year: pop[0] * np.exp(k_g * (year - 1900))
euler_integration=np.zeros(len(time))
euler_integration[0]=pop[0]
for i in range(0, len(time)-1):
    euler_integration[i+1]=euler_integration[i]+ k_g*euler_integration[i]*(time[1]-time[0])

plt.plot(time, analytical_solution(time), color = 'red', label='analytical_solution');
plt.plot(time, euler_integration, color = 'black',label='euler_intergration');
```

#3b). Decreasing the time steps will not allow for the solution to completeley converge to the actual world population because the growth rate of 0.013 is already very small. If we were to decrease the time step, then we would barely see a change in the growth. The population itself increases dramatically over many years so it would be hard to see for a smaller amount of time.

+++

__d.__ As the number of time steps increases, the Euler approximation approaches the analytical solution, not the measured data. The best-case scenario is that the Euler solution is the same as the analytical solution.

+++

2. In the freefall example you used smaller time steps to decrease the **truncation error** in our Euler approximation. Another way to decrease approximation error is to continue expanding the Taylor series. Consider the function f(x)

    $f(x)=e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\frac{x^4}{4!}+...$

    We can approximate $e^x$ as $1+x$ (first order), $1+x+x^2/2$ (second order), and so on each higher order results in smaller error. 
    
    a. Use the given `exptaylor` function to approximate the value of exp(1) with a second-order Taylor series expansion. What is the relative error compared to `np.exp(1)`?
    
    b. Time the solution for a second-order Taylor series and a tenth-order Taylor series. How long would a 100,000-order series take (approximate this, you don't have to run it)
    
    c. Plot the relative error as a function of the Taylor series expansion order from first order upwards. (Hint: use method (4) in the comparison methods from the "Truncation and roundoff error accumulation in log-log plot" figure)

```{code-cell} ipython3
from math import factorial
def exptaylor(x,n):
    '''Taylor series expansion about x=0 for the function e^x
    the full expansion follows the function
    e^x = 1+ x + x**2/2! + x**3/3! + x**4/4! + x**5/5! +...'''
    if n<1:
        print('lowest order expansion is 0 where e^x = 1')
        return 1
    else:
        ex = 1+x # define the first-order taylor series result
        for i in range(1,n):
            ex+=x**(i+1)/factorial(i+1) # add the nth-order result for each step in loop
        return ex
        
```

```{code-cell} ipython3
#d2a) Relative Error Between Approximation and Calculation
approximation = exptaylor(1,2)
calculation = np.exp(1)

print(approximation)
print(calculation)

relative_error = ((approximation - calculation)/calculation)
relative_error = np.abs(relative_error * 100)
print(relative_error)
```

```{code-cell} ipython3
#d2b) Time the Solution for Second-Order Taylor Series
%time
series1 = exptaylor(1, 2)
series2 = exptaylor(1, 10)
#For 100,000
series3 = (series2 - series1)/(10 - 2)
hundreth_series_approx = series3 * 100000
print(series1)
print(series2)
print(hundreth_series_approx)
```

```{code-cell} ipython3
#d2c) 
sample = np.arange(1, 20, 1)
length = len(sample)

list1 = []
for i in range(length):
    data_fromsample = exptaylor(1, sample[i])
    list1.append(data_fromsample)
array_approx = np.array(list1)

list2 = []
for i in range(length):
    list2.append(np.exp(1))

array_actual = np.array(list2)
    

plt.plot(sample, array_approx, color = 'red', label='analytical_solution')
plt.plot(sample, array_actual, color = 'black',label='actual_solution')
```
