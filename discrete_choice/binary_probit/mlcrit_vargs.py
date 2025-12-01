# Import modules
import numpy as np 
from scipy import stats

# Define a function that computes the log-likelihood
def mlcrit_vargs(betas, *args):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the log-likelihood for maximum likelihood estimation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.

    INPUT:
    - betas <- (k-by-1) vector of parameters to be estimated.
    - args  <- (tuple) additional arguments passed to the criterion function.
        -- yobs <-- (nobs-by-1) vector of observations of the dependent variable.
        -- xobs <-- (nobs-by-k) matrix of explanatory variables.
    
    OUTPUT:
    - llike <- (scalar) value of the log-likelihood multiplied by -1.
    ----------------------------------------------------------------------------
    '''

    # Arguments passed to the function
    vargs = args[0]
    yobs = vargs[0]
    xobs = vargs[1]

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

    return llike