#===============================================================================
# PROGRAM: Estimation of a binary probit model
#
# AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin)
#
# THIS VERSION: July 2024
#
# DESCRIPTION: This program estimates the parameters of a binary probit
# model explaining whether a car is foreign based on its weight and
# mileage, using data from http://www.stata-press.com/data/r13/auto. The
# model is Pr(foreign = 1) = Phi(beta_0 + beta_1*weight + beta_2*mpg).
#===============================================================================

# Import modules
import numpy as np
import pandas as pd
import estimagic as em
from scipy import stats
from scipy import optimize
from scipy.optimize import minimize

# Set seed
np.random.seed(13)

# Start
print('')
print('ESTIMATION OF A BINARY PROBIT MODEL')
print('')

#-------------------------------------------------------------------------------

# Import the data using a dataframe
auto = pd.read_csv('auto.csv')

# Organize the data and add a constant
choice = auto['foreign'].to_numpy()
mpg = auto['mpg'].to_numpy()
weight = auto['weight'].to_numpy()
constant = np.ones(len(choice))
regressors = np.column_stack((mpg, weight, constant))

#-------------------------------------------------------------------------------

# Define the log-likelihood function
def bprobit_llike(betas, yobs, xobs, info):
    # INPUT:
    #   yobs <-- (nobs-by-1) vector of observations of the dependent variable,
    #            i.e., indicator of discrete choice (0,1).
    #   xobs <-- (nobs-by-k) matrix of explanatory variables.
    #   betas <- (k-by-1) vector of parameters to be estimated.
    #
    # OUTPUT:
    #   llike <- (scalar) value of the log-likelihood.

    # Compute the conditional choice probabilities
    xb = np.matmul(xobs,betas)
    pxb = stats.norm.cdf(xb)

    # Substitute zero probabilities by very low probabilities to avoid 
    # convergence problems
    pxb[pxb == 0] = 0.00001

    # Substitute probabilities of one by very high probabilities to
    # avoid convergence problems
    pxb[pxb == 1] = 0.99999

    # Compute the log-likelihood
    llike = np.dot(yobs,np.log(pxb)) + np.dot((1-yobs),np.log(1-pxb))
    llike = -llike

    # Display information as the algorithm iterates
    if info['Nfeval']%10 == 0:
        print('{0:4d}   {1: 3.6f}'.format(info['Nfeval'], llike))
    info['Nfeval'] += 1

    return llike

#-------------------------------------------------------------------------------

# Initial values of the parameter vector
b0 = np.zeros(3)

# Estimate the model using the Nelder-Mead algorithm (gradient-free)
print('Estimate the model using the Nelder-Mead algorithm')
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin = minimize(bprobit_llike, b0, args=(choice, regressors, {'Nfeval':0}), \
    method='nelder-mead', options={'xatol': 1e-5, 'disp': True, 'return_all': False})

print('')
print('Parameter estimates using SciPy Nelder-Mead algorithm: ')
print('mpg:    ', outmin.x[0])
print('weight: ', outmin.x[1])
print('cons:   ', outmin.x[2])
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using Nelder-Mead's estimagic. Install estimagic by following  
# these instructions: https://estimagic.org/en/latest/getting_started/installation.html
print('Estimate the model using Nelder-Mead estimagic')
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin_em = em.minimize(
    criterion=bprobit_llike,
    params=np.zeros(3),
    algorithm="scipy_neldermead",
    criterion_kwargs={"yobs": choice, "xobs": regressors, "info": {'Nfeval':0}}
)

print('')
print('Estimation results using estimagic Nelder-Mead algorithm: ', outmin_em)
print('')
print('Parameter estimates using estimagic Nelder-Mead algorithm: ', outmin_em.params)
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using the differential evolution algorithm in SciPy
# (gradient-free). It does not perform well 
bounds_de = [(-1, 1), (-1, 1), (-10, 10)]
print('Estimate the model using the differential evolution algorithm in SciPy: ')
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin_desp = optimize.differential_evolution(bprobit_llike, bounds_de, 
    args=(choice, regressors, {'Nfeval':0}))

print('')
print('Parameter estimates using the differential evolution algorithm in SciPy: ')
print('mpg:    ', outmin_desp.x[0])
print('weight: ', outmin_desp.x[1])
print('cons:   ', outmin_desp.x[2])
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using the differential evolution algorithm in estimagic.
# The algorithm does not perform well unless appropriate bounds are provided
info = np.iinfo(np.int64)
lower_bounds_de = np.array([-1, -1, -10]) 
upper_bounds_de = np.array([1, 1, 10]) 

print('Estimate the model using the differential evolution algorithm in estimagic: ')
outmin_em = em.minimize(
    criterion=bprobit_llike,
    params=np.zeros(3),
    algorithm="scipy_differential_evolution",
    criterion_kwargs={"yobs": choice, "xobs": regressors, "info": {'Nfeval':0}},
    lower_bounds=lower_bounds_de,
    upper_bounds=upper_bounds_de
)

print('')
print('Estimation results using estimagic differential evolution algorithm: ', outmin_em)
print('Parameter estimates using estimagic differential evolution algorithm: ', outmin_em.params)
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using the Powell algorithm in SciPy (gradient-free). It 
# performs quite well and takes 434 function evaluations to converge (some more
# than Nelder-Mead)
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
print('Estimate the model using the Powell algorithm in SciPy: ')
outmin = minimize(bprobit_llike, b0, args=(choice, regressors, {'Nfeval':0}), \
    method='Powell', options={'disp': True})

print('')
print('Parameter estimates using SciPy Powell algorithm: ')
print('mpg:    ', outmin.x[0])
print('weight: ', outmin.x[1])
print('cons:   ', outmin.x[2])
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using a dual annealing algorithm. It does not perform 
# well unless appropriate bounds are provided. Given the same bounds, it does
# better than differential evolution, although it takes three times more
# iterations to converge
bounds_sa = [(-1, 1), (-1, 1), (-10, 10)]
print('Estimate the model using the dual annealing algorithm in SciPy: ')
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin_sa = optimize.dual_annealing(bprobit_llike, bounds_sa, 
    args=(choice, regressors, {'Nfeval':0}))

print('')
print('Parameter estimates using a dual annealing algorithm: ')
print('mpg:    ', outmin_sa.x[0])
print('weight: ', outmin_sa.x[1])
print('cons:   ', outmin_sa.x[2])
print('')
print('-----------------------------------------------------------------------')
print('')