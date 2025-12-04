import numpy as np
from probit_loglik_i import probit_loglik_i

def numerical_score_i(beta, y_i, x_i, h):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the numerical score for an individual observation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.
    ----------------------------------------------------------------------------
    '''

    # Convert beta to a numpy array
    beta = np.asarray(beta, dtype=float)
    k = beta.size
    score = np.zeros(k)

    for jj in range(k):
        e = np.zeros(k)
        e[jj] = 1.0

        # Compute the perturbed parameter vectors
        beta_plus = beta + h * e
        beta_minus = beta - h * e

        # Compute the function values at the perturbed parameters
        f_plus = probit_loglik_i(beta_plus, y_i, x_i)
        f_minus = probit_loglik_i(beta_minus, y_i, x_i)

        # Compute the numerical score for parameter jj
        score[jj] = (f_plus - f_minus) / (2 * h)

    return score