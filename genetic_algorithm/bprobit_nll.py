# Import modules
import numpy as np
from scipy import stats

def bprobit_nll(betas, yobs, xobs):
    """
    Compute the negative log-likelihood for a binary probit regression model.

    Parameters
    ----------
    betas : array-like
        Coefficient vector for the probit model.
    yobs : array-like
        Observed binary outcomes (0 or 1).
    xobs : array-like
        Matrix of explanatory variables (features).

    Returns
    -------
    float
        The negative log-likelihood value for the given parameters.

    Notes
    -----
    - Uses the cumulative distribution function (CDF) of the standard normal distribution
      to model the probability of the binary outcome.
    - Applies numerical clipping to probabilities for stability.
    """
    xb = xobs @ betas

    # numerical safety: clip away from 0 and 1
    p = stats.norm.cdf(xb)
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)

    # negative log-likelihood (to minimize)
    nll = -(yobs * np.log(p) + (1 - yobs) * np.log(1 - p)).sum()
    return float(nll)