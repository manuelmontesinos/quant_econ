# Import modules
import numpy as np
from scipy import stats

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

    # Record the objective function value and parameter values
    info['nfeval'].append(info['Nfeval'])
    info['fval'].append(llike)
    info['params'].append(betas.copy())

    # Display information as the algorithm iterates
    if info['Nfeval']%10 == 0:
        print('{0:4d}   {1: 3.6f}'.format(info['Nfeval'], llike))
    info['Nfeval'] += 1

    return llike