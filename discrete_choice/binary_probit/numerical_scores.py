import numpy as np
from numerical_score_i import numerical_score_i

def numerical_scores(beta, *args):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute numerical scores for maximum likelihood estimation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.
    ----------------------------------------------------------------------------
    '''

    # Arguments passed to the function
    vargs = args[0]
    yobs = vargs[0]
    xobs = vargs[1]

    # Convert to arrays
    y = np.asarray(yobs).ravel()
    X = np.asarray(xobs)
    n, k = X.shape

    # Increment for numerical derivatives
    h = 1e-6

    # Compute individual numerical scores
    S = np.zeros((n, k))
    for ii in range(n):
        S[ii, :] = numerical_score_i(beta, y[ii], X[ii, :], h)

    return S