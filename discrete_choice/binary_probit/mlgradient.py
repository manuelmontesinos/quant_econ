# Import modules
import numpy as np

# Define a function to compute the gradient for maximum likelihood estimation
def mlgradient(betas, fargs, f):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the gradient vector of numerical first derivatives for
    maximum likelihood estimation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.

    INPUT:
    - betas <- (k-by-1) vector of parameters to be estimated.
    - fargs <- (tuple) additional arguments passed to the criterion function.
    - f <------ (function) criterion function.

    OUTPUT:
    - gradvec <- (k-by-1) gradient vector of first derivatives.
    ----------------------------------------------------------------------------
    '''

    # Take the vector of estimated parameters
    betas.astype(float)

    # Number of parameters
    K = betas.shape[0]

    # Initial value of the criterion function
    f0 = f(betas, fargs)

    # Save the partial derivatives in a list
    gradvec = []
    for kk in range(K):

        # Increment of the parameter value to compute the partial derivative
        eps = abs(betas[kk]) * 1e-5
        betas0 = 1. * betas[kk]
        betas[kk] = betas[kk] + eps

        # New value of the criterion function
        f1 = f(betas, fargs)

        # Compute the partial derivative and store it in the list
        gradvec.append((np.array([f1 - f0])).item() / eps)
        betas[kk] = betas0
        
    # Reshape the gradient
    gradvec = np.array(gradvec).reshape(betas.shape)

    return gradvec