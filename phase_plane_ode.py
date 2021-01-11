from scipy import *

from pylab import *

import sys

 

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

import matplotlib.patches as mpatches





import matplotlib.animation as animation

from scipy.integrate import odeint

from numpy import arange

from pylab import *



import math 

from math import e





import scipy.integrate as spint

import pandas as pd


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 

from mpl_toolkits.axes_grid1.inset_locator import mark_inset









def xdotydotsystem(state, t):

    xdot, ydot = state

    

    

xvalues, yvalues = meshgrid(arange(0,2.5, 0.2), arange(-6, 7, 0.2))


t=0

n = 2

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)





def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,10,100000)

s0 = np.array([0,6])

s = spint.odeint(f,s0,t)



plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')



def f(x3):

   return (- w*(w-1)*x3 - x3**n)/(2*w-1) #(w(w-1)*x+x**n)/(1-2*w))#



def g(x1):

   return -(w-1)*x1 #-w*z#



def h(x2):

   return -w*x2 #w(w-1)*z#



x1 = np.arange(0.0, 2.4, 0.01)

x2 = np.arange(0.0, 2.4, 0.01)

x3 = np.arange(0.0, 2.4, 0.01)

plot(x1,g(x1), color='black', lw=1.5, label='Y-direction')

plot(x3, f(x3),color= 'green', lw=1.5, label='y-nullcline')

plot(x2,h(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')

plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)

plt.show()

# the main axes is subplot(111) by default

a = plt.axes([0.2,0.6, 0.6,0.6], facecolor='w')

#plt.plot(t,s,f(x1),f(x2),f(x3))

plot(s[:,0],s[:,1], color='red')

plot(x1,g(x1), color='black')

plot(x3, f(x3), color= 'green')

plot(x2,h(x2),color='orange')

plt.xlim(0, 2)

plt.ylim(-2,2)

plt.xticks([])

plt.yticks([])





plt.show()










xvalues, yvalues = meshgrid(arange(0,2.5, 0.2), arange(-14.5, 13.5, 0.2))

t=0

n = 3

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)


def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,3.7,10000)

s0 = np.array([0,2.5])

s = spint.odeint(f,s0,t)


plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')




def f(x3):

   return (- w*(w-1)*x3 - x3**n)/(2*w-1) #(w(w-1)*x+x**n)/(1-2*w))#



plot(x3,f(x3),color= 'green', lw=1.5, label='$\.y$-nullcline')



def g(x1):

   return -(w-1)*x1 #-w*z#



plot(x1,g(x1), 'k-', lw=1.5, label='Y-direction')



def h(x2):

   return -w*x2 #w(w-1)*z#



plot(x2,g(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')


x1 = np.arange(0.0, 2.4, 0.01)

x2 = np.arange(0.0, 2.4, 0.01)

x3 = np.arange(0.0, 2.4, 0.01)


plot(x1,g(x1), color='black', lw=1.5, label='Y-direction')

plot(x3, f(x3),color= 'green', lw=1.5, label='$\.y$-nullcline')

plot(x2,h(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')

plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)

plt.show()

# the main axes is subplot(111) by default

a = plt.axes([0.2,0.6, 0.6,0.6], facecolor='w')

#plt.plot(t,s,f(x1),f(x2),f(x3))

plot(s[:,0],s[:,1], color='red')

plot(x1,g(x1), color='black')

plot(x3, f(x3), color= 'green')

plot(x2,h(x2),color='orange')

plt.xlim(0, 2)

plt.ylim(-2,2)

plt.xticks([])

plt.yticks([])





plt.show()







xvalues, yvalues = meshgrid(arange(0,2.5, 0.2), arange(-8.5, 7, 3))

t=0

n = 4

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)





def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,2.7,100)

s0 = np.array([0,1.7])

s = spint.odeint(f,s0,t)



plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')



def f(x3):

   return ( w*(1-w)*x3 - x3**n)/(2*w-1) #(w(w-1)*x+x**n)/(1-2*w))#

x3 = np.arange(0.0, 1.3, 0.01)

plot(x3,f(x3),color= 'green', lw=1.9, label='$\.y$-nullcline')



def g(x1):

   return -(w-1)*x1 #-w*z#

x1 = np.arange(0.0, 2.41, 0.01)

plot(x1,g(x1), 'k-', lw=1.5, label='Y-direction')



def h(x2):

   return -w*x2 #w(w-1)*z#

x2 = np.arange(0.0, 2.41, 0.01)

plot(x2,g(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')



plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)


x1 = np.arange(0.0, 2.4, 0.01)

x2 = np.arange(0.0, 2.4, 0.01)

x3 = np.arange(0.0, 1.36, 0.01)


plot(x1,g(x1), color='black', lw=1.5, label='Y-direction')

plot(x3, f(x3),color= 'green', lw=1.5, label='x-nullcline')

plot(x2,h(x2), color='orange', lw=1.5, label='X-direction')



plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)

plt.show()

# the main axes is subplot(111) by default

a = plt.axes([0.2,0.6, 0.6,0.6], facecolor='w')

#plt.plot(t,s,f(x1),f(x2),f(x3))

plot(s[:,0],s[:,1], color='red')

plot(x1,g(x1), color='black')

plot(x3, f(x3), color= 'green')

plot(x2,h(x2),color='orange')

plt.xlim(0, 2)

plt.ylim(-2,2)

plt.xticks([])

plt.yticks([])


plt.show()





xvalues, yvalues = meshgrid(arange(0,1.2, 0.2), arange(-2, 2, 0.2))

t=0

n = 5

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)





def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,9.88,100)

s0 = np.array([0,0.11])

s = spint.odeint(f,s0,t)



plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')



def f(x):

   return ( w*(1-w)*x - x**n)/(2*w-1) #(w(w-1)*x+x**n)/(1-2*w))#

x = np.arange(0.0, 1, 0.01)

plot(x,f(x),color= 'green', lw=1.9, label='x-nullcline')



def f(x1):

   return -(w-1)*x1 #-w*z#

x1 = np.arange(0.0, 1, 0.01)

plot(x1,f(x1), 'k-', lw=1.5, label='Y-direction')



def f(x2):

   return -w*x2 #w(w-1)*z#

x2 = np.arange(0.0, 1, 0.01)

plot(x2,f(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')





plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)



plt.show()





xvalues, yvalues = meshgrid(arange(0,2.5, 0.2), arange(-7.5, 8, 0.2))

t=0

n = 6

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)





def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,2.42,100)

s0 = np.array([0,1])

s = spint.odeint(f,s0,t)



plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')



def f(x3):

   return (- w*(1-w)*x3 + x3**n)/-(2*w-1) 







def g(x1):

   return -(w-1)*x1 #-w*z#






def h(x2):

   return -w*x2 #w(w-1)*z#




legend(loc='upper right')

x1 = np.arange(0.0, 2.4, 0.01)

x2 = np.arange(0.0, 2.4, 0.01)

x3 = np.arange(0.0, 1.10999, 0.01)

plot(x1,g(x1), color='black', lw=1.5, label='Y-direction')

plot(x3, f(x3),color= 'green', lw=1.5, label='$\.y$-nullcline')

plot(x2,h(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')

plt.xlabel('z', fontsize=16)

plt.ylabel('y', fontsize=16)

plt.show()

# the main axes is subplot(111) by default

a = plt.axes([0.2,0.6, 0.6,0.6], facecolor='w')

#plt.plot(t,s,f(x1),f(x2),f(x3))

plot(s[:,0],s[:,1], color='red')

plot(x1,g(x1), color='black')

plot(x3, f(x3), color= 'green')

plot(x2,h(x2),color='orange')

plt.xlim(0, 2)

plt.ylim(-2,2)

plt.xticks([])

plt.yticks([])





plt.show()







xvalues, yvalues = meshgrid(arange(0,2.5, 0.2), arange(-8.5, 7, 3))

t=0

n = 3

w = 2/(n-1)

xdot = yvalues

ydot = -(2*w - 1)*yvalues - w*(w-1)*xvalues - xvalues**n

streamplot(xvalues, yvalues, xdot, ydot)





def f(s,t):

   xdot = s[1]

   ydot =  -(2*w - 1)*s[1] - w*(w-1)*s[0] - s[0]**n

   return np.array([xdot,ydot])

t = np.linspace(0,13.7,100)

s0 = np.array([0,1])

s = spint.odeint(f,s0,t)



plt.plot(s[:,0],s[:,1], color='red',lw=2.5, label='sol. curve')



def f(x3):

   return ( w*(1-w)*x3 - x3**n)/(2*w-1) #(w(w-1)*x+x**n)/(1-2*w))#

x3 = np.arange(0.0, 1.3, 0.01)

plot(x3,f(x3),color= 'green', lw=1.9, label='$\.y$-nullcline')



def g(x1):

   return -(w-1)*x1 #-w*z#

x1 = np.arange(0.0, 2.41, 0.01)

plot(x1,g(x1), 'k-', lw=1.9, label='Y-direction')



def h(x2):

   return -w*x2 #w(w-1)*z#

x2 = np.arange(0.0, 2.41, 0.01)

plot(x2,g(x2), color='orange', lw=1.5, label='X-direction')

legend(loc='upper right')
