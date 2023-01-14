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

# Project 1 By: Zyaja Huggan

```{code-cell} ipython3
#all modules needed for this assignment
import numpy as np #math functions
import matplotlib.pyplot as plt #plotting
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
#1
current_temp = 85 #current temperature
future_temp = 74 #temperature after two hours
T_a = 65 #ambient temperature
change_in_temp = (future_temp - current_temp) #change in temperature
time_elasped = 2 #change in time
K = -(change_in_temp/(time_elasped*(current_temp - T_a))) #formula to solve for empirical constant from the equations above
K
```

The empirical constant is 0.275.

+++

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
#2
def empirical_constant_K(current_temp, future_temp, ambient_temp, time_elasped): 
    '''Calculates empirical constant
    
    Arguments
    ---------
    current_temp : Temperature of the corpse at t=0hours
    future_temp  : Temperature hours later
    ambient_temp : Constant ambient temperature
    time_elasped : Change in time
    
    Returns
    -------
    empirical_constant(K) 
    
    '''
    return -((future_temp - current_temp)/(time_elasped*(current_temp - T_a))) #formula to solve for empirical constant from the equations above
print(empirical_constant(85, 74, 65, 2))
    
```

The empirical constant is 0.275.

+++

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
#3a
K = 0.275 #empirical constant
T_0 = 85 #temperature at zero hours
T_a = 65 #ambient temperature
t = np.arange(0, 24, 1) #time range represents 24 hour day period, step time 1
T_t = np.linspace(T_0, T_0, len(t)) #create array of temperatures
dt = t[1] - t[0] #change in time 
for i in range(1, len(t)):
    T_t[i] = T_t[i-1] - (K*(T_t[i-1]-T_a)*dt) #solving for temperatures
```

```{code-cell} ipython3
def analytical_solution(T_0,T_a,K,t):
    '''Calculates the final temperature
    
    Arguments
    ---------
    T_0 : Temperature of the corpse at t=0hours
    T_a : Constant ambient temperature
    K   : Empirical Constant 
    t   : Time
    
    Returns
    -------
    T_t : Final Temperature
    
    
    '''
    T_t = T_a + (T_0 - T_a)*(np.exp(-K*t)) #first-order thermal system equation
    return T_t

plt.plot(t, analytical_solution(T_0,T_a,K,t), label = "Analytical") #plotting analytical solution
plt.plot(t, T_t, label = "Euler-Numerical ") #plotting numerical solution
plt.xlabel("Time") #x-axis label
plt.ylabel("Temperature")#y-axis label
plt.title('Convergence of Euler Intergration')
plt.legend()
plt.show()
```

```{code-cell} ipython3
t = np.arange(0, 24, 0.7) #time range represents 24 hour day period, step time 0.7
T_t = np.linspace(T_0, T_0, len(t)) #create array of temperatures
dt = t[1] - t[0] #change in time 
for i in range(1, len(t)):
    T_t[i] = T_t[i-1] - (K*(T_t[i-1]-T_a)*dt) #solving for temperatures

plt.plot(t, analytical_solution(T_0,T_a,K,t), label = "Analytical") #plotting analytical solution
plt.plot(t, T_t, label = "Euler-Numerical ") #plotting numerical solution
plt.xlabel("Time") #x-axis label
plt.ylabel("Temperature")#y-axis label
plt.title('Convergence of Euler Intergration')
plt.legend()
plt.show()
```

```{code-cell} ipython3
t = np.arange(0, 24, 0.5) #time range represents 24 hour day period, step time 0.5
T_t = np.linspace(T_0, T_0, len(t)) #create array of temperatures
dt = t[1] - t[0] #change in time 
for i in range(1, len(t)):
    T_t[i] = T_t[i-1] - (K*(T_t[i-1]-T_a)*dt) #solving for temperatures

plt.plot(t, analytical_solution(T_0,T_a,K,t), label = "Analytical") #plotting analytical solution
plt.plot(t, T_t, label = "Euler-Numerical ") #plotting numerical solution
plt.xlabel("Time") #x-axis label
plt.ylabel("Temperature")#y-axis label
plt.title('Convergence of Euler Intergration')
plt.legend()
plt.show()
```

The Euler Integration converges to the analytical solution as the time step is decreased.

+++

# 3b

+++

In order to find the temperature as t appraches infinity, the analytical function can be used or the ending number of the data can be found as seen below.

```{code-cell} ipython3
#3b
print(analytical_solution(T_0,T_a,K,np.inf)) #final temperature as time approaches infinity
```

or

```{code-cell} ipython3
T_t[-1] #last item in the array/data
```

The final temperature as t approaches infinity is 65 degrees F.

+++

# 3c

+++

The time of death can be found both analytically using a function and graphically, as seen below. 

```{code-cell} ipython3
#3c
def time(T_a,T_0,K, T_t):
    '''Calculates time at a certain temperature
    
    Arguments
    ---------
    T_0 : Temperature of the corpse at t=0hours
    T_a : Constant ambient temperature
    K   : Empirical Constant 
    T_t : Current Temperature 
    
    Returns
    -------
    t   : Time
    
    
    '''
    t = (np.log((T_t - T_a)/(T_0 - T_a)))/(-K) #rearranging first-order thermal equation to solve for time
    return t 
print(time(T_a, T_0, K, 98.7))  #time when temperature is 98.7 degrees F(time of death)
```

The temperature at the time of death was about 1.9 hours before the end of the day so that would be at a little after 10pm. We can see this graphically below:

```{code-cell} ipython3
K = 0.275 #empirical constant
t = np.linspace(0, 2)
dt = t[1] - t[0] #change in time

T_t_f = np.zeros(len(t))
T_t_b = np.zeros(len(t))
T_t_f[0] = 85 #temp at zero
T_t_b[0] = 85 #temp at zero

for i in range (1, len(t)):
    T_t_f[i] = T_t_f[i-1] - (K*dt*(T_t_f[i-1]-65)) #graphs the data forwards
    T_t_b[i] = T_t_b[i-1] + (K*dt*(T_t_b[i-1]-65)) #graphs the data backwards
plt.plot(t, T_t_f, label = 'Forwards')
plt.plot(-t, T_t_b, label = 'Backwards')
plt.title('Time of Death')
plt.ylabel('Temperature')
plt.xlabel('Time')
plt.legend()
plt.axhline(y = 98.7, color ='b', linestyle ='-') #horizontal line to indicate where the temperature is, 98.7 degrees F

```

This graphical approach matches the analytical approach answer of about -1.9, a little after 10pm.
