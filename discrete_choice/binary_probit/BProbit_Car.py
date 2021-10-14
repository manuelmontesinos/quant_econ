#===============================================================================
# PROGRAM: Estimation of a binary probit model
#
# AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and Barcelona
# School of Economics)
#
# THIS VERSION: October 2021
#
# DESCRIPTION: This program estimates the parameters of a binary probit
# model explaining whether a car is foreign based on its weight and
# mileage, using data from http://www.stata-press.com/data/r13/auto. The
# model is Pr(foreign = 1) = Phi(beta_0 + beta_1*weight + beta_2*mpg).
#===============================================================================

# Import modules
import numpy as np
import pandas as pd
from scipy import stats
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

# Estimate the model using the Nelder-Mead algorithm
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin = minimize(bprobit_llike, b0, args=(choice, regressors, {'Nfeval':0}), \
    method='nelder-mead', options={'xatol': 1e-5, 'disp': True, 'return_all': False})

print('Parameter estimates: ')
print('mpg:    ', outmin.x[0])
print('weight: ', outmin.x[1])
print('cons:   ', outmin.x[2])
