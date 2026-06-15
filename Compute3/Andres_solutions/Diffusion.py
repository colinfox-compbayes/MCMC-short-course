# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:46:58 2016

@author: Andrés Christen
"""


from robfem import robfem

from pytwalk import pytwalk

from numpy import array, linspace, where, intersect1d, pi, sqrt, log, zeros, matrix, arange
from scipy.integrate import odeint
from scipy.stats import norm, gamma
from scipy.linalg import inv
from pylab import plot, imshow, figure, subplot, title, ylim, ylabel, xlim, xlabel


def G( u, t, A): ##rhs of ode
    
    Gu = array(A * matrix(u).T).reshape(u.shape[0])
    Gu[0] = 0.0
    Gu[-1] = 0.0 ### Boundary conditions
    return Gu



N = 75  ## #Divisions in x
a = 0
b = 10
x = linspace( a, b, N+1)
al = 5.0
alpha = array([al]*N) ##Alpha constant

x1 = 7.5
x2 = 8.0
u0 = zeros(N+1) #linspace( 7.0, 2.0, N+1)
u0[intersect1d( where( x >= x1 ) , where( x <= x2 )  )] = 1.0
M = 75  ## # divisions in t
tt = linspace( 0.0, 2.0, M)


def FM( D ):
    """The forward map maps the diffusion coeficient to the
       state space of u(x,t) = sol."""

    K, M = robfem( x, alpha, D)
    A = inv(M)*K
    sol = odeint( G, u0, tt, args=(A,))
    return sol
    
### True values:
d_true = 1.0
D_true = array([-d_true]*N)
sol_true = FM( D_true)
    


def SynthData( sol, indx, sigma):
    """Produce synthetic data for solution sol, at x[indx]'s with std dev. sigma.
       observetions are at the last time."""
    
    return sol_true[-1,obs_indx] + sigma*norm.rvs(size=len(obs_indx))

obs_indx = arange(1,N,3) ##We take observations every three x points
n = len(obs_indx)
obs_x = x[obs_indx]
obs_sigma = 0.04
obs_var = obs_sigma**2
obs_const = -log(obs_sigma*sqrt(2*pi))

obs_data = SynthData( sol_true, obs_indx, obs_sigma)


def PlotSol( sol, image=False, data=None):
    """Plot solution sol, if image is not False, plot 3D color map in figure(image[0])
       and t[0] and t[-1] solutions in figure(image[0])."""

    if (image != False):
        figure(image[0])
        imshow(sol, origin='lower', extent=[ x[0], x[-1], tt[0], tt[-1]], aspect='auto' )
        xlabel(r"$x$")
        ylabel(r"$t$")
        [plot( [0,10], [t,t], '-', color='grey', linewidth=0.3) for t in tt[1::10]]
        [plot( [0,10], [t,t], '-', color='grey', linewidth=0.3) for t in tt[::10]]
        figure(image[1])

    plot( x, sol[0,:], 'k-', linewidth=2)
    plot( x, sol[-1,:], 'g-', linewidth=2)
    if (data != None):
        plot( data[0], data[1], 'ko')
    ylim((0,ylim()[1]*1.2))
    ylabel(r"$C$")
    xlabel(r"$x$")

#PlotSol( sol_true, image=[0,1], data=[ obs_x, obs_data])


def LogLikelihood1( d, s2, const, data):
    """Univariate likelihood, one single value for the diffusion coef."""
    
    sol = FM(array([-d]*N))
    return const - (0.5/s2)*sum((data-sol[-1,obs_indx])**2)

#### Gamma prior for d:
al0 = 3.0
be0 = 3.0 ###Expected value is 1, the true value

def LogPrior1( d, alpha, beta):
    
    return (alpha-1.0)*log(d) - beta*d

#### Functions for the twalk:    
def Supp1(x):
    return ( x[0] > 0.0)

def Energy1(x):
    
    d = x[0]
    
    return -1.0*(LogLikelihood1( d, s2=obs_var, const=obs_const, data=obs_data)\
                + LogPrior1( d, alpha=al0, beta=be0))

def Initd1():
    """Initial values distribution, sample from the prior."""
    return gamma.rvs( al0, scale=1.0/be0, size=1)
    

Twalk1 = pytwalk( n=1, U=Energy1, Supp=Supp1)

#Twalk1.Run( T=10000, x0=Initd1(), xp0=Initd1())
#Twalk1.RunRWMH( T=10000, x0=Initd1(), sigma=array([0.2]))

def AnaMCMC(start=100):
    figure(1)
    Twalk1.Ana()

    figure(2)
    Twalk1.Hist(par=0, start=start, xlab=r"$d$", normed=True)
    MAP = Twalk1.Output[where(Twalk1.Output[:,1] == min(Twalk1.Output[:,1]))[0][0],0]
    plot( [MAP,MAP], [0,0.5*gamma.pdf( MAP, al0, scale=1/be0)], 'r-')
    dd = linspace( 0.0, xlim()[1], 100)
    plot( dd, gamma.pdf( dd, al0, scale=1/be0), 'g-')

    figure(3)
    D_MAP = array([-MAP]*N)
    sol_MAP = FM( D_true)
    imshow(sol_MAP, origin='lower', extent=[ x[0], x[-1], tt[0], tt[-1]], aspect='auto' )
    xlabel(r"$x$")
    ylabel(r"$t$")
    [plot( [0,10], [t,t], '-', color='grey', linewidth=0.3) for t in tt[1::10]]
    [plot( [0,10], [t,t], '-', color='grey', linewidth=0.3) for t in tt[::10]]
    plot( [0,10], [tt[10],tt[10]], '-', color='pink', linewidth=0.5)
    plot( [0,10], [tt[11],tt[11]], '-', color='pink', linewidth=0.5)

    figure(4)
    plot( x, sol_MAP[0,:], 'k-')
    plot( x, sol_MAP[-1,:], 'r-')
    plot( obs_x, obs_data, 'ko')
    ylim((0,ylim()[1]*1.2))
    ylabel(r"$o^C$")
    xlabel(r"$x$")
    ### iid sample of size 100
    print "Thinning every %d iterations." % ((Twalk1.T-start)/100,)
    print "MAP (red):", MAP
    print "Predictive at t=0 (black), t=%f (pink) and last t (grey)." % (tt[10],)
    pseudo_iid = Twalk1.Output[arange( start, Twalk1.T-start, (Twalk1.T-start)/100),0]
    for d in pseudo_iid:
        D = array([-d]*N)
        sol = FM(D)
        plot( x, sol[-1,:], '-', color='grey')
        plot( x, sol[10,:], '-', color='pink')
    plot( x, sol_MAP[-1,:], 'r-')



