# -*- coding: utf-8 -*-
"""
@author: digiovanniyani
"""
#HW1--TASK1------------------------------------------------

from scipy import *
import numpy as np
from numpy import array
from scipy import integrate

"""
#HOMEWORK1---TASK2------------------------------------------------------------
a=0
b=1
#n=5000
h=(b-a)/(n)

def f(x):
    return x**2

 

def trapezoidal(f,a,b,n):
    return (h/2)*(f(a)+f(b))+h*( sum(f(a+h*i)  for i in range(1, (n-1))) )
    
   
    
#print(trapezoidal(f,a,b,(n-1)))
#print(trapezoidal(f,a,b,(n)))
#lambda x: f(x)
#print(integrate.quad(f, 0, 1))


for n in range (4000,6000):
    x1=trapezoidal(f,a,b,(n-1))
    x2=trapezoidal(f,a,b,n)


    if abs(x1-x2) < 0.002:
        print(x1,x2)
        break

print(trapezoidal(f,a,b,n=4000))

lambda x: f(x)
print(integrate.quad(f, 0, 1))
"""

"""
a=0
b=1
n=500
h=(b-a)/(n)

def f(x):
    return x**2

 

def trapezoidal(f,a,b,n):
    return (h/2)*(f(a)+f(b))+h*( sum(f(a+h*i)  for i in range(1, (n-1))) )
    
   
    
print(trapezoidal(f,a,b,n))


lambda x: f(x)
print(integrate.quad(f, 0, 1))
"""
"""
#HOMEWORK1---TASK2---------????---------------------------------------------------

a=0
b=1
#n=5000
h=(b-a)/(n)

def f(x):
    return x**2

 

def trapezoidal(f,a,b,n):
    return (h/2)*(f(a)+f(b))+h*( sum(f(a+h*i)  for i in range(1, (n-1))) )
    
   
    
#print(trapezoidal(f,a,b,(n-1)))
#print(trapezoidal(f,a,b,(n)))
#lambda x: f(x)
#print(integrate.quad(f, 0, 1))


for n in range (4000,6000):
    x1=trapezoidal(f,a,b,(n-1))
    x2=trapezoidal(f,a,b,n)


    if abs(x1-x2) < 0.002:
        print(x1,x2)
        break

print(trapezoidal(f,a,b,n=4000))  #larger values of n return a strange outcome

lambda x: f(x)
print(integrate.quad(f, 0, 1))

"""

#HW1--TASK1------------------------------------------------
"""
from scipy import *
import numpy as np
from numpy import array
from scipy import integrate


a=0
b=1
n=5
h=(b-a)/(n)

def f(x):
    return x**2

 

def trapezoidal(f,a,b,n):
    return (h/2)*(f(a)+f(b))+h*( sum(f(a+h*i)  for i in range(1, (n-1))) )
    
   
    
print(trapezoidal(f,a,b,n))


lambda x: f(x)
print(integrate.quad(f, 0, 1))
"""
"""
#HOMEWORK-1--TASK-3-------------------------------------------------
from scipy import integrate
import matplotlib.pyplot as plt
 
 
a=0
b=1
n=500
h=(b-a)/(n)
 
def f(x):
    return x**2
 
def trapezoidal(f,a,b,n):
    return (h/2)*(f(a)+f(b))+h*( sum(f(a+h*i)  for i in range(1, (n-1))))
 
print(trapezoidal(f,a,b,n))
 
# Why this?
lambda x: f(x)
 
c= integrate.quad(f,0,1)
print(c)
 
error_list=[abs(trapezoidal(f,a,b,n)-c[0]) for n in range (300,900)]
print(error_list)
plt.plot(error_list, [(b-a)/(n) for n in range(300, 900)])
plt.show()

"""

#HOMEWORK-1-----TASK-4-----------------------------------------------------------------
from scipy import *
import numpy as np
from numpy import array
from scipy import integrate
import matplotlib.pyplot as plt




r=5.0
A=1000.0
n=3
balance=10000

def f(balance,r,A,n):
        
     
        for year in range(n):
            balance=(balance)*(r/100.0 +1)- A
            if balance <=0:
                return 0
        return balance
        
print(round(f(balance,r,A,n),0))

    
def loan_reminder(principal, annual_interest_rate, payment, years):
    #principal = loan
    for year in range(years):                           # repeat process for every year
        #interest = principal * annual_interest_rate     # calculate interest accrued 
        principal = principal*(annual_interest_rate + 1)  - payment      # calculate new principal
        if principal <= 0:                              # check whether loan is repaid
            return 0                                    # if loan repaid return 0 as loan reminder
    return principal                                    # return loan reminder after years
 
print(round(loan_reminder(10000, 0.05, 1000, 6),3))
#7237.184375000001


"""
#HOMEWORK-1-----TASK-5-----------------------------------------------------------------
def f(balance,r,A,n):
        r=5.0
        A=1000.0
        n=20
        balance=10000
        L=[]
          
        for year in range(n):
            balance=(balance)*(r/100.0 +1)- A
            L.append(balance)
            if balance <=0:
                balance=0
                break
        return L
    
                
        
print(f(balance,r,A,n))
print('the loan is fully paid back', len(f(balance,r,A,n)) ,'years later')
#print(L,n) #should print the last non negative value and the iterations
            #needed to get there

"""