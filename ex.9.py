

from scipy import *
from pylab import*
import numpy as np
from numpy import array
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy.optimize import fsolve
from math import log
import pickle
import sys
import itertools 

file=open('kwh.dat','r')
yearmonthday=[]; kwh1=[]
for i in file:
    tempsplit=i.split( )
    yearmonthday.append(tempsplit[0])
 
 
    kwh1.append(int(tempsplit[1])) # or kwh1.append((tempsplit[1]))
#print(kwh1)
#print(yearmonthday)

yearmonthday.reverse()
kwh1.reverse()

(yearmonthday.reverse())
(kwh1.reverse())

kwh=np.array(kwh1)
kwhmonthly=diff(kwh)

print(kwh1)
print(yearmonthday)
#print(kwh.shape)
#print(kwhmonthly.shape)

#x=range(1,len(yearmonthday))
#plt.plot(x,kwhmonthly)


maxmonth=max(kwh) #maximum value
minmonth=min(kwh) #minimum value
#print(maxmonth)   
#print(minmonth)

kwh_index = np.argmax(kwh) #position of the maximum value
#print(kwh_index)
#print(len(kwh))       
#print(yearmonthday[kwh_index]) #date position



res =[(kwh[i+1]-kwh[i]) for i in range(len(kwh)-1)]
res1=[kwh[i]-kwh[i+1] for i in range (len(kwh)-1)]
print(max(res))
print(max(res1))
print(max(-kwhmonthly))
print(min(res1))
print(min(-kwhmonthly))
res1_index=np.argmax(res1)
kmon_index=np.argmax(-kwhmonthly)
print(res1_index)
print(kmon_index)
print(yearmonthday[res1_index])
print(yearmonthday[kmon_index])
print(mean(kwh))
