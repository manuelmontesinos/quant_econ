import numpy as np
from scipy import stats

def probit_loglik_and_scores(beta, yobs, xobs):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: Compute the log-likelihood and scores for a binary probit model
    for each observation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.

    INPUT:
    - beta  <- (k,) vector of parameters.
    - yobs  <- (n,) vector of observations of the dependent variable.
    - xobs  <- (n, k) matrix of explanatory variables.

    OUTPUT:
    - ll_i  <- (n,) vector of per-observation log-likelihoods.
    - scores <- (n, k) matrix of per-observation scores.
    ----------------------------------------------------------------------------
    '''

    # Convert inputs to numpy arrays. yobs is flattened to a 1-D array, X 
    # becomes an (n, k) matrix, beta is a (k,) vector
    beta = np.asarray(beta)
    y = np.asarray(yobs).ravel()
    X = np.asarray(xobs)

    # Conditional choice probabilities. z has shape (n,), Phi is the probit
    # probability and phi is the standard normal density at z, needed for the
    # scores
    z = X @ beta
    Phi = stats.norm.cdf(z)
    phi = stats.norm.pdf(z)

    # Avoid log(0) by bounding probabilities away from 0 and 1
    eps = 1e-12
    Phi = np.clip(Phi, eps, 1 - eps)

    # Per-observation log-likelihood
    ll_i = y * np.log(Phi) + (1 - y) * np.log(1 - Phi)

    # Per-observation score factors (shape (n,)). This is the derivative
    # of ll_i with respect to the scalar index z_i
    score_factor = y * (phi / Phi) - (1 - y) * (phi / (1 - Phi))

    # Each row i: s_i(beta) = score_factor[i] * x_i' (shape (n, k)). This is
    # to convert the scalar derivative into a gradient with respect to each
    # parameter in beta. Multiplying by X produces an (n, k) matrix of scores
    scores = score_factor[:, None] * X

    return ll_i, scores