import numpy as np
from scipy import stats

def probit_loglik_i(beta, y_i, x_i):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the log-likelihood for an individual observation in a
    binary probit model.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.
    ----------------------------------------------------------------------------
    '''

    # Compute the conditional choice probabilities
    xb_i = np.matmul(x_i, beta)
    pxb_i = stats.norm.cdf(xb_i)

    # Avoid log(0)
    eps = 1e-12
    pxb_i = np.clip(pxb_i, eps, 1 - eps)

    if y_i == 1:
        return np.log(pxb_i)
    else:
        return np.log(1 - pxb_i)