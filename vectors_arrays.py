# -*- coding: utf-8 -*-



#TASK1------------------------------------------------------------------

"""
import numpy as np
from numpy import array

def isSymmetric(mat,N): 
 
  newmat = [[0,0,0],[0,0,0],[0,0,0]]
  for i in range(N): 
    for j in range(N): 
      newmat[i][j]=mat[j][i]
  return newmat

def isSkew(mat,N):
  newmat = [[0,0,0],[0,0,0],[0,0,0]]
  for i in range(N):
    for j in range(N):
        newmat[i][j] = -mat[j][i]
  return newmat

def test(mat,N):
  if isSymmetric(mat,N)==mat:
    return 1
  if isSkew(mat,N)==mat:
    return -1
  else:
    return 0;

N=3
mat = [ [0,1,-2 ], [ -1,0,3], [2,-3,0 ] ] 
print(test(mat,N))


N=3
mat = [ [1,1,-1 ], [ 1,2,0], [-1,0,5 ] ] 
print(test(mat,N))


N=3
mat = [ [1,1,-1 ], [ 11,2,0], [4,9,5 ] ] 
print(test(mat,N))
"""
"""
#TASK2----------------------------------------------------------------------

import numpy as np
from numpy import array
from scipy.linalg import*

def v_vector(V):
  ##  return v /=np.linalg.norm(v)
    return (np.array(V))
def w_vector(W):
    return (np.array(W))

def test_ort(V,W):  
    DotProduct = np.dot(V,W)
    if DotProduct == 0:
        return True
    else:
        return False
    
    
V=[1,0,-1]
W=[1,sqrt(2),1]
print(test_ort(V,W))
"""

"""
#TASK3part1-------------------------------------------------------------------

import numpy as np
from numpy import array

V=[1,2,3]

n_vec=sqrt(V[0]**2+V[1]**2+V[2]**2)

    
def v_vector(V):
    return ((V[0]/n_vec)+(V[1]/n_vec)+(V[2]/n_vec))

print(v_vector(V))
"""
"""

#TASK3part2-----------------------------------------------------------
from scipy.linalg import norm

V=[1,2,3]

def v_vector(V):
    return (linalg.norm(V))

print(sum(V/v_vector(V)))
"""
"""

#TASK4------?-------------------------------------------------------------
import numpy as np
from numpy import array
theta = np.radians(92)
c, s = np.cos(theta), np.sin(theta)
M=array([[c,-s],[s,c]])
def matrix():
    return M/M.transpose()

print(matrix())
"""
#if lower != none:
#            self.lower=lower
#        else:
#            self.lower=upper
#            
#        if not isinstance((self.upper, numbers.Real) | (self.lower, numbers.Real) ):
#            raise TypeError("only real numbers allowed")
#        elif self.lower< self.upper:
#            raise ValueError("left endpoint cannot be larger than right endpoint")
#TASK5/6-----------?------------------------------
import numpy as np
from numpy import array
import scipy.linalg 


z = np.zeros((20,20))
i,j=np.indices(z.shape)
z[i==j]=4
z[i==j+1]=1 #=-1 for task 6
z[i==j-1]=1
eigenvalues=np.linalg.eigvals(z)
print(z)
print(eigenvalues)
