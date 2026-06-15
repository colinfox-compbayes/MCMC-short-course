# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:49:12 2016

@author: Andres Christen, copied from Colin Fox Matlab code
"""

from numpy import matrix, zeros, linspace, arange, array, ix_

###Example:

N = 5
a = 0
b = 10
x = linspace( a, b, N+1)
al = 3.0
alpha = array([al]*N) ##Alpha constant
D = linspace( 1.0, 5.0, N)

def robfem( x, alpha, D ):
    """ Make stiffness matrix via FEM discretization for
        differential operator -Du'' and alpha u
        over finite domain [a,b]

        input:
            x       N+1 vector of x values sorted in ascending order
            alpha   N vector of alpha values in each interval (x_i,x_i+1)
            D       N vector of D values in each (x_i,x_i+1)
        output:
            K       (N+1)*(N+1) matrix containing stiffness matrix
            M       (N+1)*(N+1) matrix containing mass matrix

        adapted from femprec
        cf. notes of 29/9/07
    """

    npt = len(x)

    K = matrix( zeros((npt,npt)) ) 
    M = matrix( zeros((npt,npt)) ) 

    # set local to global node mapping for each element
    nodenum = zeros((npt-1,2), dtype=int)
    nodenum[:,0] = range( npt-1)# easy in this case
    nodenum[:,1] = range( 1, npt)# easy in this case

    # calculate lengths    
    el_len = x[nodenum[:,1]] - x[nodenum[:,0]]
    
    nel = len(el_len)# number of elements

    ls1 = (1/6.0)*matrix([[2, 1],[1, 2]]) # local mass matrix 
    ls2 = (1.0)*matrix([[1, -1],[-1, 1]])# local stiffness matrix 

    for elcnt in range(nel):
        le = el_len[elcnt]    # element length
        ae = alpha[elcnt]   # element alpha value
        ce = D[elcnt]    # element D value
        # u^2 and u'^2 terms
        M[ix_(nodenum[elcnt,:],nodenum[elcnt,:])] += (ae * le) * ls1
        K[ix_(nodenum[elcnt,:],nodenum[elcnt,:])] += (ce / le) * ls2
        
    return K, M


"""
print "x=\n", x
print "D=\n", D
print "alpha=\n", alpha
K, M =robfem( x, alpha, D)
print "K, M = robfem( x, alpha, D) =\n", K, "\n", M
"""
