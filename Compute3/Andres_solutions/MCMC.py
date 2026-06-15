# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:06:46 2016

@author: jac
"""



#
#  twalktutorial.py
#  
#  Examples for the twalk implementation in Python,
#  Created by J Andres Christen, jac at cimat.mx .
#
#  Check the current version in the file VERSION.
#
#  See http://www.cimat.mx/~jac/twalk/ for more details.
#


import pytwalk

from numpy import ones, zeros, log, array
from numpy.random import uniform

from pylab import figure

## You may get the inline help:
# pytwalk.pytwalk?
# pytwalk.pytwalk.Run?


##################################################################
### This sets a MCMC for n=1 independent normals (default)


def NormU(x):
	"""-log of a objective, Gaussian N(0,1):"""
	return 0.5*x[0]**2

def NormSupp(x):
	return True


Normal = pytwalk.pytwalk( n=1, U=NormU, Supp=NormSupp)

## This runs the twalk


### This does a Random Walk Metriopolis Hastings with 5000 iterations
### initial point x0 and standar dev's for the normal jumps = sigma

Normal.RunRWMH( T=100000, x0=1.0*ones(1), sigma=0.05*ones(1))


### This will do a basic output analsis

figure(1)
Normal.Ana()
#Normal.Save("Exptwalk.dat")  ### Saves MCMC output
#Normal.Output ### This is the MCMC outpour

### This plots the histogram of the parameter, with burn-in of start=100
figure(2)
Normal.Hist( par=0, start=100)



### This runs instead the t-walk:  No sigma, only two initial values.  More in pytwalktutorial.py
#Normal.Run( T=5000, x0=1.0*ones(1), xp0=0.1*ones(1))


