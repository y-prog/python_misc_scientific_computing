# -*- coding: utf-8 -*-
"""
Yani Di Giovanni
"""

from scipy import *
import numpy as np
from numpy import array
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy.optimize import fsolve
from math import log
import matplotlib as mpl
import itertools
from matplotlib.pyplot import*


"""
#TASK-1---------------------------------------------------------------------
def applog(n,x):
    a0=(1+x)/2
    b0=sqrt(x)
    for i in range(n):
        a0=(a0+b0)/2
        b0=sqrt((a0)*b0)
    return (x-1)/a0
 
"""


"""
#TASK-2-------------------------------------------------------------------------------------
def applog(n,x):
    a0=(1+x)/2
    b0=sqrt(x)
   
    for i in range(n):
        a0=(a0+b0)/2
        b0=sqrt((a0)*b0)
      
        
    return (x-1)/a0

print("the log approximation is: " , ((applog(4,4)))) #trial
print("log value-------------------:" , (np.log(4)) ) #trial
                   
error=(abs( applog(4,4)- (np.log(4))   ) )                    
print("the error is : ", error)
  
x = np.linspace(10, 500, 100)
nn=range(1,6)
colors = mpl.cm.rainbow(np.linspace(0, 1, len(nn)))


for c,n in zip(colors,nn):
    plt.plot(x, (applog(n,x))) 
plt.plot(x, np.log(x), color='red', label='ln(x)') 
plt.legend(loc='upper left')
plt.show()   
for c,n in zip(colors,nn):       
    plt.plot(x,abs(np.log(x)-applog(n,x)) )
plt.show()  
"""      


"""
#TASK-3-------------------------------------------------------------------


def applog(n,x):
    a0=(1+x)/2
    b0=sqrt(x)
   
    for i in range(n):
        a0=(a0+b0)/2
        b0=sqrt((a0)*b0)
    return (x-1)/a0


list1=range(1,6)
for nn in list1:
    error=abs((applog(nn,1.41)- np.log(1.41)))
    print(error)
    plt.plot(nn,error,'-or')
"""  
"""
#TASK-4-----------------------------------------------------------------------------------
def fastln(n, x):
    a =(1 + x) / 2
    g = x ** (1 / 2)
    d = [[a]] 
    for i in range(1, n + 1):
        a = (a + g) / 2
        g = ((a * g) ** (1 / 2))
        d[-1].append(a) #this line states that i is iterated before k
    for k in range(1, n + 1):
        d.append([])
        for i in range(n + 1):
            d[-1].append((d[k - 1][i] - 2 ** (-2 * k) * d[k - 1][i - 1])/(1 - 2 ** (-2 * k)))
    return (x - 1)/d[-1][-1]
   
print(fastln(4,4))
print(math.log(4, math.e))
print(fastln(5,4))
"""

"""
#TASK-5----------------------------------------------------------------
def fastln(n, x):
    a =(1 + x) / 2
    g = x ** (1 / 2)
    d = [[a]] 
    for i in range(1, n + 1):
        a = (a + g) / 2
        g = ((a * g) ** (1 / 2))
        d[-1].append(a)
    for k in range(1, n + 1):
        d.append([])
        for i in range(n + 1):
            d[-1].append((d[k - 1][i] - 2 ** (-2 * k) * d[k - 1][i - 1])/(1 - 2 ** (-2 * k)))
    return (x - 1)/d[-1][-1]
   

x = np.linspace(0,20, 1000)
nn=range(2,6)
print(type(nn))

colors = mpl.cm.rainbow(np.linspace(0, 1, len(nn)))

for c,n in zip(colors,nn):    
    plt.plot(x,abs(np.log(x)-fastln(n,x)) , label="$n ={0}$".format(n) )
    yscale('log')
    ylim(10**-19,10**-5)
plt.legend(loc='upper right')
plt.ylabel('error')
plt.xlabel('x')
plt.show()
"""

