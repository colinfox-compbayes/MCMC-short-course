#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 02:49:57 2022

@author: colin
"""

import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib as plt
import math
import scipy.io

# constants from makefake
npix = 100              # make npix*npix image
cellrad = 9.5           # radius of cells
stddev = 0.1            # noise standard deviation

def goodbad_mcmc(N):
    """Input positive integer N nr steps. """

    im = Image.open('slide.tif')
    Dslide = np.array(im, dtype='float32')/255.  # this is taken as data. Dividing by 255 undoes the scale when writing a tif to UINT8
    
    # mat_dict = scipy.io.loadmat('slide.mat')   # to input the .mat file
    # Dslide = mat_dict["slide"]
    
    plotaxis = plt.pyplot.imshow(Dslide, cmap='gray') # show data in plot window
    plt.pyplot.show()
    
    MoveRatio = np.array([1, 1, 1, 1,], dtype='float32') 
    MoveProb = np.cumsum(MoveRatio[0:-1]/sum(MoveRatio))
    
    showeach = 50           # subsampling for plotting state

    mean_num = 5.           # my guess at mean number of points
    pgeo = 1/(1+mean_num)   # parameter in geometric distribution prior over number
    lampois = mean_num      # for Poisson distribution prior
    
    X = []                  # starting state is null state (state is list of 3-lists)
    #Xlist = [] # a list of list states
    llold = loglike(X,Dslide)
    
    geo_prior = True  # True for geometric prior, False for Poisson
    if geo_prior:
        lpold = logprior_geo(X,pgeo)
    else:
        lpold = logprior_pois(X,lampois)  
    
    # initialize outputs here
    prop = np.zeros(len(MoveRatio))     # collect number of proposals
    acc = np.zeros(len(MoveRatio))      # collect acceptance ratios
    kvec = np.zeros(N+1)                # collect marginal state over number
    kvec[0] = len(X)
    
    llvec = np.zeros(N+1) 
    llvec[0] = llold
    lpvec = np.zeros(N+1) 
    lpvec[0] = lpold
    
    # here is the MCMC     
    for iter in range(0, N):
        kernum = sum(MoveProb < np.random.uniform()) + 1    # choose a kernel (1 to #Moves)
        match str(kernum):
            case '1':       # birth/death
                Xp, lh = Simh1(X)
            case '2':       # flip label
                Xp, lh = Simh2(X)
            case '3':       # move point
                Xp, lh = Simh3(X)
            case '4':       # swap two marked points
                Xp, lh = Simh4(X)
    
        llnew = loglike(Xp,Dslide)
        #print('llnew',llnew)
        if geo_prior:
            lpnew = logprior_geo(Xp,pgeo)
        else:
            lpnew =  logprior_pois(X,lampois)
            
        lalpha = llnew + lpnew - llold - lpold + lh
        #print('lpold=',lpold,' lpnew=',lpnew,' llold=',llold,' llnew=',llnew,' lh=',lh)
        #print('lalpha',lalpha)
        
        lalpha = min(0. , lalpha) # protect against math range error in math.exp
        if (np.random.uniform() < math.exp(lalpha)):
            #print('MH:accept')
            #accepted
            X = Xp.copy()
            llold = llnew
            lpold = lpnew
            
        # accumulate stats
        #print('proposal=',Xp,' state X=',X)
        kvec[iter+1] = len(X)
        llvec[iter+1] = llold
        lpvec[iter+1] = lpold
        if (iter % showeach) == 0:      # show the current state
            plotstate(X) # need to work out how to replace plot
    
    return kvec, X, llvec, lpvec
    

#############################################################################

def loglike(X,D):
    """Calculate log likelihood for state X (up to an additive constant independent of state)"""

    ll = (-1/(2*stddev**2))*np.sum(np.square(makeimage(X,npix) - D))
# note that writing to tiff clips the original observations to [0,1], 
# hence this is not quite the correct likelihood.
# compare hist(slide(:),100) before and after writing to .tif to see the clipping
      
    return ll

#############################################################################

def logprior_geo(X,pgeo):
    """Calculate log geometric prior for state X """
    
    k = len(X)
    if k >= 0:
        lp = k*math.log(1-pgeo) + math.log(pgeo)
    else:
        lp = -math.inf

    return lp

#############################################################################

def logprior_pois(X,lampois):
    """Calculate log Poisson prior for state X """
    
    k = len(X)
    if k >= 0:
        lp = k*math.log(lampois) - lampois - math.log(math.factorial(k))
    else:
        lp = -math.inf

    return lp

#############################################################################

def Simh1(X):
    """Simulate kernel 1, return proposed state and log Hastings ratio"""
    # birth/death (chosen with equal prob)
    
    k = len(X)
    # Xp = X.copy()
    Xp = [xi.copy() for xi in X] # deep-ish copy
    
    if np.random.uniform() < 0.5:
        #print('Simh1:birth')
        # birth -- pick a random place from 1 to k+1, and insert a valid marked point
        inew = np.random.randint(0,k+1)      # insert new object before index inew, or at end if inew = k+1
        xynew = np.random.randint(0,npix,(2))
        lnew = np.random.randint(0,2,(1))          # pick a random mark from the prior

        # make a 3-list to insert
        mparr = np.concatenate([xynew, lnew])
        Xp.insert(inew,mparr.tolist())             # insert new marked point
        if Xp[inew][0] < 0 or  Xp[inew][1] < 0:
            print('holy fuck Xp=',Xp)
    else:
        #print('Simh1:death')
        # death -- uniformly at random pick a marked point and zap it
        if k != 0:
            izap = np.random.randint(0,k)
            del Xp[izap]        # remove marked point
            
    lh = 0.      # this proposal is symmetric
    return Xp,lh

#############################################################################

def Simh2(X):
    """Simulate kernel 2, return proposed state and log Hastings ratio"""
    # pick a label uniformly at random, and flip it
    
    k = len(X)
    # Xp = X.copy()
    Xp = [xi.copy() for xi in X] # deep-ish copy
    #print('Simh2:flip')
    if k != 0:
        iflip = np.random.randint(0,k) # pick an index
        tmp = Xp[iflip].copy()         # new pointer to marked-point list
        tmp[2] = 1-tmp[2]              # and flip 1 <--> 0
        Xp[iflip] = tmp
         
    lh = 0.      # this proposal is symmetric
    return Xp,lh

#############################################################################

def Simh3(X):
    """Simulate kernel 3, return proposed state and log Hastings ratio"""
    # move point
    # pick a point at random, and move its center in a random window
    
    k = len(X)
    # Xp = X.copy()
    Xp = [xi.copy() for xi in X] # deep-ish copy
    #print('X before move:',X)
    #print('Simh3:move')
    if k != 0:
        w = 6                                     # window size in pixels
        imove = np.random.randint(0,k)            # pick an index
        xymove = np.random.randint(-w,w,2)        # amount to move
        tmp = X[imove].copy()          # new pointer to marked-point list
        tmp[0] = max(min(tmp[0] + xymove[0],npix-1),0)
        tmp[1] = max(min(tmp[1] + xymove[1],npix-1),0)
        Xp[imove] = tmp   # and move point
        
    #print('Xp after move:',Xp)
        
    lh = 0.   # this proposal is symetric
    
    return Xp,lh

#############################################################################

def Simh4(X):
    """Simulate kernel 4, return proposed state and log Hastings ratio"""
    # swap two marked points (so change positions in front to back ordering)
    
    k = len(X)
    # Xp = X.copy()
    Xp = [xi.copy() for xi in X] # deep-ish copy

    #print('Simh4:swap')
    
    if k >= 2:
        imove = np.random.randint(0,k,2)          # pick two indices
        tmp = Xp[imove[0]].copy()
        Xp[imove[0]] = Xp[imove[1]].copy()
        Xp[imove[1]] = tmp                         # and swap marked points
        
    lh = 0.   # this proposal is symetric

    return Xp,lh

############################################################################
def makeimage(X,npix):
    """Xslide = makeimage(X,npix)
    function file that takes a state X and produces the image (slide) with
    associated good and bad cells.

    Based on MatLab code of 23 July 2019

    State is length k list 3 vectors representing marked points: [(x1,y1,l1), ... , (xk,yk,lk)] 
    k is number of points
    each l is 0 = 'bad', 1 = 'good'
    first point is infront, last is at back """

    cellrad = 9.5           # radius of cells

    Xslide = np.ones((npix,npix)) # make npix*npix image, 1 = bright

    k = len(X)
    if  k == 0:
        return Xslide # nothing to do
    else:
        for icell in range(k-1,-1,-1): # count backwards
            #print('icell=',icell)
            if X[icell][2] == 0:
                Xslide = putbad(Xslide,X[icell][0],X[icell][1],cellrad)
            else:
                Xslide = putgood(Xslide,X[icell][0],X[icell][1],cellrad)
                
    return Xslide

############################################################################
def putbad(a,x,y,r):
    """ Put a 'bad' cell in image a """
    #print('putbad: a=',a)
    m, n = a.shape
    
    mm, nn = np.meshgrid(range(m), range(n), indexing='xy')
    
    d2 = np.square((mm - x)) + np.square((nn - y))
    a[np.where(d2 <= r**2)] = 0.5
    a[np.where(d2 <= (r/2)**2)] = 1.
    
    return a
    
############################################################################
def putgood(a,x,y,r):
    """ Put a 'bad' cell in image a """
    #print('putgood: a=',a)    
    m, n = a.shape
        
    mm, nn = np.meshgrid(range(m), range(n), indexing='xy')
        
    d2 = np.square((mm - x)) + np.square((nn - y))
    a[np.where(d2 <= r**2)] = 0.5
    a[np.where(d2 <= (r/2)**2)] = 0.
    
    return a
    
############################################################################
def showstate(X):
    """ Display state X """
    
    Xslide = makeimage(X,npix)
    Ximage = 255*Xslide
    img = Image.fromarray(Ximage.astype(np.uint8))
    img.show()
    
    return img

############################################################################
def plotstate(X):
    """ Display state X in plot window """
    
    Xslide = makeimage(X,npix)
    # if len(X) == 0:
    #     Xslide[1,1]=0. # fudge to plot empty state as white
    plt.pyplot.clf()
    # imgplot = plt.pyplot.imshow(Xslide, cmap='gray')
    plt.pyplot.imshow(Xslide, cmap='gray', vmin=0, vmax=1)
    plt.pyplot.show()
    
    # return #imgplot

############################################################################
# main line
############################################################################
      
if __name__ == "__main__":
    N = 10000
    goodbad_mcmc(N)
    
    
    
