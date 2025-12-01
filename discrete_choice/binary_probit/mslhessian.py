# Import modules
import numpy as np 

# Import functions written for the project
from mslgradient import mslgradient

# Define a function to compute the Hessian matrix for maximum likelihood 
# estimation
def mslhessian(betas, fargs, f):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the Hessian matrix of numerical second derivatives for
    maximum likelihood estimation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.

    INPUT:
    - betas <- (k-by-1) vector of parameters to be estimated.
    - fargs <- (tuple) additional arguments passed to the criterion function.
    - f <------ (function) criterion function.

    OUTPUT:
    - hessianmat <- (k-by-k) Hessian matrix of second derivatives.

    CALLS:
    - mslgradient <- function that computes the gradient vector.
    ----------------------------------------------------------------------------
    '''

    # Number of parameters
    K = betas.shape[0]

    # Define a matrix 
    hessianmat = np.zeros((K,K))

    # Initial value of the gradient
    gd_0 = mslgradient(betas, fargs, f)

    for kk in range(K):

        # Increment of the parameter value to compute the partial derivative
        eps = abs(betas[kk]) * 1e-5
        betas0 = 1. * betas[kk]
        betas[kk] = betas0 + eps
        gd_1 = mslgradient(betas, fargs, f)
        hessianmat[:,kk] = ((gd_1 - gd_0) / eps).reshape(betas.shape[0])
        betas[kk] = betas0
        print('2nd derivative computed for parameter: ', kk)

    return hessianmat