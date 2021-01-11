
from scipy import *
import numpy as np
from numpy import array
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy.optimize import fsolve
from math import log
import sys

"""
#TASK-1---------------------------------------------------------

class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
        
"""   

"""
#TASK-2/6-----------------------------------------------------------------------

class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
    def __add__(self,other):
        a,b,c,d = self.left, self.right, other.left, other.right
        return interval(a+c,b+d)
    
    def __sub__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval((a-d),(b-c))
    def __mul__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval(min(a*c, a*d, b*c, b*d),
                            max(a*c, a*d, b*c, b*d))
    def __truediv__(self, other):
        a, b, c, d = self.left, self.right, other.left, other.right
        # [c,d] cannot contain zero:
        if c*d <= 0:
            raise ValueError ('Interval %s cannot be denominator because it contains zero' % other)
        return interval(min(a/c, a/d, b/c, b/d),
                            max(a/c, a/d, b/c, b/d))
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)

I = interval
aa = I(-3,0)
bb = I(5,5)
cc = aa/bb
print(aa,bb)       
print(cc)

"""
"""
#TASK-3---------------------------------------------------------------
class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left

    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)

I = interval
aa = I(1,2)
print(aa)


"""
"""
#TASK-4-----------------------------------------------------------------

class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
    def __add__(self,other):
        a,b,c,d = self.left, self.right, other.left, other.right
        return interval(a+c,b+d)
    
    def __sub__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval((a-d),(b-c))
    def __mul__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval(min(a*c, a*d, b*c, b*d),
                            max(a*c, a*d, b*c, b*d))
    def __truediv__(self, other):
        a, b, c, d = self.left, self.right, other.left, other.right
        # [c,d] cannot contain zero:
        print(min(a/c, a/d, b/c, b/d))
        
        if c*d <= 0:
            raise ValueError ('Interval %s cannot be denominator because it contains zero' % other)
        
        return interval(min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d))
    
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)

I1 = interval(1, 4)
I2 = interval(-2, -1)
print(I1 + I2)       
print(I1-I2)
print(I1*I2)
print(I1/I2) 
"""

"""
#TASK-5-------------------------------------------------------------------------

class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
    def __add__(self,other):
        a,b,c,d = self.left, self.right, other.left, other.right
        return interval(a+c,b+d)
    
    def __sub__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval((a-d),(b-c))
    def __mul__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval(min(a*c, a*d, b*c, b*d),
                            max(a*c, a*d, b*c, b*d))
    def __truediv__(self, other):
        a, b, c, d = self.left, self.right, other.left, other.right
        # [c,d] cannot contain zero:
        print(min(a/c, a/d, b/c, b/d))
        
        if c*d <= 0:
            raise ValueError ('Interval %s cannot be denominator because it contains zero' % other)
        
        return interval(min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d))
    
    def __contains__(self,other):
    return (other>=self.left and other<=self.right)
    
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)
"""
"""
#TASK-7---------------------------------------------------

class interval:
    def __init__(self,left,right=None):
        self.right=left
        self.left=left
    def __add__(self,other):
        a,b,c,d = self.left, self.right, other.left, other.right
        return interval((a+c,b+d))
    
    def __sub__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval((a-d),(b-c))
    def __mul__(self,other):
        a,b,c,d=self.left,self.right,other.left,other.right
        return interval(min(a*c, a*d, b*c, b*d),
                            max(a*c, a*d, b*c, b*d))
    def __truediv__(self, other):
        a, b, c, d = self.left, self.right, other.left, other.right
        # [c,d] cannot contain zero:
        print(min(a/c, a/d, b/c, b/d))
        
        if c*d <= 0:
            raise ValueError ('interval %s cannot be denominator because it contains zero' % other)
        
        return interval(min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d))
    
    def __contains__(self,other):
        return (other>=self.left and other<=self.right)
    
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)
 
print(interval(1))
"""

"""
#TASK-8---------------------------------------------------------------------------------
class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
    def __add__(self,other):
        a,b = self.left + other, self.right + other
        return interval(a,b)
    
    def __radd__(self,other):
        a,b = self.left + other, self.right + other
        return interval(a,b)
    
    def __sub__(self,other):
        a,b = self.left - other, self.right - other
        return interval(a,b)
    
    def __rsub__(self,other):
        a,b = other - self.left , other - self.right 
        return interval(b,a)
    
    def __mul__(self,other):
        a,b = self.left *other, self.right *other
        return interval(a,b)
    
    def __rmul__(self,other):
        a,b = self.left *other, self.right *other
        return interval(a,b)
    
    def __truediv__(self, other):
        a, b, c, d = self.left, self.right, other.left, other.right
        # [c,d] cannot contain zero:
        print(min(a/c, a/d, b/c, b/d))
        
        if c*d <= 0:
            raise ValueError ('Interval %s cannot be denominator because it contains zero' % other)
        
        return interval(min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d))
    
    def __contains__(self,other):
        return (other>=self.left and other<=self.right)
    
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)


print(interval(2,3) + 1) # [3, 4] 
print(1 + interval(2,3) )# [3, 4] 
print(1.0 + interval(2,3)) # [3.0, 4.0] 
print(interval(2,3) + 1.0) # [3.0, 4.0]
print(1 - interval(2,3)) # [-2, -1]   
print(interval(2,3) -1) # [1, 2] 
print(1.0 - interval(2,3)) # [-2.0, -1.0]
print(interval(2,3) - 1.0) # [1.0, 2.0] 
print(interval(2,3) * 1) # [2, 3] 
print(1 * interval(2,3)) # [2, 3] 
print(1.0 * interval(2,3) )# [2.0, 3.0] 
print(interval(2,3) * 1.0 )# [2.0, 3.0]

"""


#TASK-9/10-------------------------------------------------------------------
class interval:
    def __init__(self,left,right):
        self.right=right
        self.left=left
        def check_i(other):
            if isinstance(other,float) or isinstance(other,int):
                a,b = other,  other
            elif isinstance(other,interval):
                a,b = other.left, other.right
            return a,b
        self.check_i = check_i
        
    def __add__(self,other):
        aa,bb = self.check_i(other)
        a,b = self.left + aa, self.right + bb
        return interval(a,b)
    
    def __radd__(self,other):
        aa,bb = self.check_i(other)
        a,b = self.left + aa, self.right + bb
        return interval(a,b)
    
    def __sub__(self,other):
        cc,dd = self.check_i(other)
        a,b = self.left - dd, self.right - cc
        return interval(a,b)
    
    def __rsub__(self,other):
        aa,bb = self.check_i(other)
        a,b = aa - self.left , bb - self.right 
        return interval(b,a)
    
    def __mul__(self,other):
        aa,bb = self.check_i(other)
        a,b = self.left *aa, self.right *bb
        return interval(a,b)
    
    def __rmul__(self,other):
        aa,bb = self.check_i(other)
        a,b = self.left *aa, self.right *bb
        return interval(a,b)
    
    def __truediv__(self, other):
        aa,bb = self.check_i(other)
        a, b, c, d = self.left, self.right, aa.left, bb.right
        # [c,d] cannot contain zero:
        print(min(a/c, a/d, b/c, b/d))
        
        if c*d <= 0:
            raise ValueError ('Interval %s cannot be denominator because it contains zero' % other)
        
        return interval(min(a/c, a/d, b/c, b/d), max(a/c, a/d, b/c, b/d))
    
    def __contains__(self,other):
        return (other>=self.left and other<=self.right)
    
    def __pow__(self, other):
           

        a,b=self.left, self.right
    
        if other %2 !=0:
            return interval(a**other,b**other)
        elif a>=0:
            return interval(a**other,b**other)
        elif b<0:
            return interval(b**other, a**other)
        else:
            return interval(0,max(a**other,b**other))
      
    
    def __contains__(self,other):
        return (other>=self.left and other<=self.right)
    
    def __repr__(self):
        return '[{},{}]'.format(self.left,self.right)
    
x = interval(-2,2) # [-2, 2] 
print(x**2) # [0, 4] 
print(x**3) # [-8, 8]

print(interval(2,3) + 1) # [3, 4] 
print(1 + interval(2,3) )# [3, 4] 
print(1.0 + interval(2,3)) # [3.0, 4.0] 
print(interval(2,3) + 1.0) # [3.0, 4.0]
print(1 - interval(2,3)) # [-2, -1]   
print(interval(2,3) -1) # [1, 2] 
print(1.0 - interval(2,3)) # [-2.0, -1.0]
print(interval(2,3) - 1.0) # [1.0, 2.0] 
print(interval(2,3) * 1) # [2, 3] 
print(1 * interval(2,3)) # [2, 3] 
print(1.0 * interval(2,3) )# [2.0, 3.0] 
print(interval(2,3) * 1.0 )# [2.0, 3.0]



print(-2 in interval(0,3))
#sys.exit('a')
xl = np.linspace(0.,1,1000)
xu = np.linspace(0.,1,1000) + 0.5
#print(interval(1,2)+interval(0,1))

def f(I):
    return 3*I**3 - 2*I**2 - 5*I -1

list1 = []
m=1000
for n in range(m):
	list1.append(interval(xl[n], xu[n])) 

xll=[]
xuu=[]
for i in range(m):
    I = f(list1[i])
    xll.append(I.left)
    xuu.append(I.right)
plt.plot(xl,xll)
plt.plot(xl,xuu)
plt.title(r'$p(I)=3I^3-2I^2-5I-1$, I=Interval($x,x+0.5$)')
plt.xlabel(r'$x$')
plt.ylabel(r'$p(I)$')
plt.show()
