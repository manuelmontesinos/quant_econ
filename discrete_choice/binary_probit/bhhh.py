import numpy as np

def bhhh(loglik_and_scores, beta0, yobs, xobs, maxiter, tol, step0, verbose):
    '''
    ----------------------------------------------------------------------------
    FUNCTION: BHHH optimization for maximum likelihood estimation.

    AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).

    THIS VERSION: December 2025.

    INPUT:
    - loglik_and_scores <- function that computes the log-likelihood and 
      per-observation scores.
    - beta0             <- (k,) initial parameter vector.
    - yobs              <- (n,) vector of observations of the dependent variable.
    - xobs              <- (n, k) matrix of explanatory variables.
    - maxiter           <- maximum number of iterations.
    - tol               <- tolerance for convergence based on the infinity norm
                            of the gradient.
    - step0             <- initial step size for line search.
    - verbose           <- if True, print iteration details.

    OUTPUT:
    - results <- dictionary with the following entries:
        -- beta      : (k,) estimated parameter vector.
        -- ll        : (scalar) log-likelihood at the solution.
        -- vcov      : (k, k) variance-covariance matrix of estimates.
        -- se        : (k,) standard errors of estimates.
        -- niter     : number of iterations performed.
        -- converged : boolean indicating if convergence was achieved.
    ----------------------------------------------------------------------------
    '''

    # Initial parameter vector
    beta = np.asarray(beta0, dtype=float)
    niter = 0
    converged = False

    for it in range(1, maxiter + 1):

        niter = it

        # Log-likelihood and per-observation scores at current beta
        ll_i, scores = loglik_and_scores(beta, yobs, xobs)
        ll = ll_i.sum()

        # Gradient (shape (k,)) and BHHH "Hessian" approximation (shape (k,k))
        g = scores.sum(axis=0)
        B = scores.T @ scores

        # Check convergence using the infinity norm of the gradient. Print iteration
        # if verbose. If the gradient is small, we have converged
        grad_norm = np.linalg.norm(g, ord=np.inf)
        if verbose:
            print(f"Iter {it:3d}: ll = {ll: .6f}, ||grad||_inf = {grad_norm: .3e}")

        if grad_norm < tol:
            converged = True
            break

        # Compute search direction: d = B^{-1} g. This is a Newon-like step using 
        # the BHHH approximation to the Hessian
        try:
            direction = np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            # If B is singular or ill-conditioned, fall back to pseudo-inverse
            direction = np.linalg.pinv(B) @ g

        # Line search to ensure increase in log-likelihood
        step = step0
        ll_current = ll
        while step > 1e-8:
            beta_new = beta + step * direction
            ll_i_new, _ = loglik_and_scores(beta_new, yobs, xobs)
            ll_new = ll_i_new.sum()
            if ll_new >= ll_current:
                # If the new value of the log-likelihood is not worse than 
                # the current value, accept the step and update beta. Otherwise,
                # halve the step size and try again
                beta = beta_new
                ll = ll_new
                break
            # Backtrack
            step *= 0.5

        # If the step gets very small, give up on improving
        if step <= 1e-8:
            if verbose:
                print("Line search failed to improve objective; stopping.")
            converged = False
            break

    # Final loglik and scores at solution (for vcov)
    ll_i, scores = loglik_and_scores(beta, yobs, xobs)
    ll = ll_i.sum()
    B = scores.T @ scores

    # Variance-covariance matrix: inverse of BHHH information matrix
    try:
        vcov = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(B)

    # Standard errors
    se = np.sqrt(np.diag(vcov))

    # Return everyting in a dictionary
    return {
        "beta": beta,
        "ll": ll,
        "vcov": vcov,
        "se": se,
        "niter": niter,
        "converged": converged,
    }