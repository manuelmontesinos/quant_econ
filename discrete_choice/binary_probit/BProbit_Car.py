#===============================================================================
# PROGRAM: Estimation of a binary probit model
#
# AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin)
#
# THIS VERSION: July 2024
#
# DESCRIPTION: This program estimates the parameters of a binary probit
# model explaining whether a car is foreign based on its weight and
# mileage, using data from http://www.stata-press.com/data/r13/auto. The
# model is Pr(foreign = 1) = Phi(beta_0 + beta_1*weight + beta_2*mpg).
#===============================================================================

# Import modules
import numpy as np
import pandas as pd
import estimagic as em
import matplotlib.pyplot as plt

from scipy import stats
from scipy import optimize
from scipy.optimize import minimize

from bprobit_llike import bprobit_llike
from mlcrit_vargs import mlcrit_vargs
from mlhessian import mlhessian
from mlgradient import mlgradient
from numerical_scores import numerical_scores
from bhhh import bhhh
from probit_loglik_and_scores import probit_loglik_and_scores

# Set seed
np.random.seed(13)

# Start
print('')
print('ESTIMATION OF A BINARY PROBIT MODEL')
print('')

#-------------------------------------------------------------------------------

# Import the data using a dataframe
auto = pd.read_csv('auto.csv')

# Organize the data and add a constant
choice = auto['foreign'].to_numpy()
mpg = auto['mpg'].to_numpy()
weight = auto['weight'].to_numpy()
constant = np.ones(len(choice))
regressors = np.column_stack((mpg, weight, constant))

# Initial values of the parameter vector
b0 = np.zeros(3)

#-------------------------------------------------------------------------------

# # Estimate the model using the Nelder-Mead algorithm (gradient-free)
# print('Estimate the model using the Nelder-Mead algorithm')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin = minimize(bprobit_llike, b0, args=(choice, regressors, history), \
#     method='nelder-mead', options={'xatol': 1e-5, 'disp': True, 'return_all': False})

# print('')
# print('Parameter estimates using SciPy Nelder-Mead algorithm: ')
# print('mpg:    ', outmin.x[0])
# print('weight: ', outmin.x[1])
# print('cons:   ', outmin.x[2])
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Estimate the model using Nelder-Mead's estimagic. Install estimagic by following  
# # these instructions: https://estimagic.org/en/latest/getting_started/installation.html
# print('Estimate the model using Nelder-Mead estimagic')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin_em = em.minimize(
#     criterion=bprobit_llike,
#     params=np.zeros(3),
#     algorithm="scipy_neldermead",
#     criterion_kwargs={"yobs": choice, "xobs": regressors, "info": history}
# )

# print('')
# print('Estimation results using estimagic Nelder-Mead algorithm: ', outmin_em)
# print('')
# print('Parameter estimates using estimagic Nelder-Mead algorithm: ', outmin_em.params)
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Estimate the model using the differential evolution algorithm in SciPy
# # (gradient-free). It does not perform well 
# bounds_de = [(-1, 1), (-1, 1), (-10, 10)]
# print('Estimate the model using the differential evolution algorithm in SciPy: ')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin_desp = optimize.differential_evolution(bprobit_llike, bounds_de, 
#     args=(choice, regressors, history))

# print('')
# print('Parameter estimates using the differential evolution algorithm in SciPy: ')
# print('mpg:    ', outmin_desp.x[0])
# print('weight: ', outmin_desp.x[1])
# print('cons:   ', outmin_desp.x[2])
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Estimate the model using the differential evolution algorithm in estimagic.
# # The algorithm does not perform well unless appropriate bounds are provided
# info = np.iinfo(np.int64)
# lower_bounds_de = np.array([-1, -1, -10]) 
# upper_bounds_de = np.array([1, 1, 10])

# # Trying with wider bounds, but it does not perform well
# # lower_bounds_de = np.array([-1000, -1000, -1000]) 
# # upper_bounds_de = np.array([1000, 1000, 1000])

# print('Estimate the model using the differential evolution algorithm in estimagic: ')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# outmin_em = em.minimize(
#     criterion=bprobit_llike,
#     params=np.zeros(3),
#     algorithm="scipy_differential_evolution",
#     criterion_kwargs={"yobs": choice, "xobs": regressors, "info": history},
#     lower_bounds=lower_bounds_de,
#     upper_bounds=upper_bounds_de
# )

# print('')
# print('Estimation results using estimagic differential evolution algorithm: ', outmin_em)
# print('Parameter estimates using estimagic differential evolution algorithm: ', outmin_em.params)
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Define lists to store the number of iterations and the value of the objective
# # function
# itlist = []
# llikelist = []

# # Estimate the model using the Powell algorithm in SciPy (gradient-free). It 
# # performs quite well and takes 434 function evaluations to converge (some more
# # than Nelder-Mead)
# print('Estimate the model using the Powell algorithm in SciPy: ')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin = minimize(bprobit_llike, b0, args=(choice, regressors, history), \
#     method='Powell', options={'disp': True})

# # Plot the values of the criterion function
# plt.plot(history['nfeval'], history['fval'], linestyle='-', color='b')
# plt.xlabel('Iterations')
# plt.ylabel('Log-likelihood')
# plt.title('Values of the log-likelihood function (SciPy Powell)')
# plt.show()

# print('')
# print('Parameter estimates using SciPy Powell algorithm: ')
# print('mpg:    ', outmin.x[0])
# print('weight: ', outmin.x[1])
# print('cons:   ', outmin.x[2])
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Estimate the model using the Powell algorithm in estimagic (gradient-free)
# print('Estimate the model using the Powell algorithm in estimagic: ')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin_em = em.minimize(
#     criterion=bprobit_llike,
#     params=np.zeros(3),
#     algorithm="scipy_powell",
#     criterion_kwargs={"yobs": choice, "xobs": regressors, "info": history}
# )

# # # Make a plot of the evolution of the criterion function
# # fig = em.criterion_plot(outmin_em)
# # fig.show()

# # # Make a plot of the evolution of the parameter estimates
# # fig = em.params_plot(outmin_em)
# # fig.show()

# print('')
# print('Estimation results using estimagic Powell algorithm: ', outmin_em)
# print('Parameter estimates using estimagic Powell algorithm: ', outmin_em.params)
# print('')
# print('-----------------------------------------------------------------------')
# print('')

# #-------------------------------------------------------------------------------

# # Estimate the model using a dual annealing algorithm. It does not perform 
# # well unless appropriate bounds are provided. Given the same bounds, it does
# # better than differential evolution, although it takes three times more
# # iterations to converge

# bounds_sa = [(-1, 1), (-1, 1), (-10, 10)]

# # Trying with wider bounds, but it does not perform well
# # bounds_sa = [(-1000, 1000), (-1000, 1000), (-1000, 1000)]

# print('Estimate the model using the dual annealing algorithm in SciPy: ')
# history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
# print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
# outmin_sa = optimize.dual_annealing(bprobit_llike, bounds_sa, 
#     args=(choice, regressors, history))

# print('')
# print('Parameter estimates using a dual annealing algorithm: ')
# print('mpg:    ', outmin_sa.x[0])
# print('weight: ', outmin_sa.x[1])
# print('cons:   ', outmin_sa.x[2])
# print('')
# print('-----------------------------------------------------------------------')
# print('')

#-------------------------------------------------------------------------------

# Estimate the model using the L-BFGS-B algorithm in SciPy
print('Estimate the model using the L-BFGS-B algorithm in SciPy: ')
history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
outmin = minimize(bprobit_llike, b0, args=(choice, regressors, history), \
    method='L-BFGS-B', options={'disp': True})
estcoefs = outmin.x

# Plot the values of the criterion function
plt.plot(history['nfeval'], history['fval'], linestyle='-', color='b')
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.title('Values of the log-likelihood function (SciPy L-BFGS-B)')
plt.show()

# Compute the standard errors by inverting the Hessian matrix and taking
# square roots of the diagonal elements
ml_args = (choice, regressors)
hess = mlhessian(estcoefs, ml_args, mlcrit_vargs)
invhess = np.linalg.inv(hess)
std_errors = np.sqrt(np.diag(invhess))

print('')
print('Parameter estimates and standard errors using SciPy L-BFGS-B algorithm: ')
print(f"mpg:    {outmin.x[0]:.4f} ({std_errors[0]:.4f})")
print(f"weight: {outmin.x[1]:.4f} ({std_errors[1]:.4f})")
print(f"cons:   {outmin.x[2]:.4f} ({std_errors[2]:.4f})")
print('')
print('-----------------------------------------------------------------------')
print('')

# Norm of the gradient at the estimates (should be close to zero)
grad = mlgradient(estcoefs, ml_args, mlcrit_vargs)
print("Gradient at the estimates:", grad)
norm_grad = np.linalg.norm(grad)
print("Gradient norm at the estimates:", norm_grad)

# Per-observation scores at the estimates (shape (n,k))
S = numerical_scores(estcoefs, ml_args)
print("Per-observation scores at the estimates: ", S)

# BHHH information matrix: sum of outer products (shape (k,k))
bhhh_info = S.T @ S
print("BHHH information matrix: ", bhhh_info)

# Var-cov matrix and standard errors using BHHH
vcov_bhhh = np.linalg.inv(bhhh_info)
std_errors_bhhh = np.sqrt(np.diag(vcov_bhhh))
print("Variance-covariance matrix using BHHH: ", vcov_bhhh)

print('')
print('Parameter estimates and standard errors given by the BHHH estimate: ')
print(f"mpg:    {outmin.x[0]:.4f} ({std_errors_bhhh[0]:.4f})")
print(f"weight: {outmin.x[1]:.4f} ({std_errors_bhhh[1]:.4f})")
print(f"cons:   {outmin.x[2]:.4f} ({std_errors_bhhh[2]:.4f})")
print('')
print('-----------------------------------------------------------------------')
print('')

#-------------------------------------------------------------------------------

# Estimate the model using the BHHH optimization routine
results = bhhh(loglik_and_scores=probit_loglik_and_scores,
    beta0=b0,
    yobs=choice,
    xobs=regressors,
    maxiter=300,
    tol=1e-4,
    step0=1.0,
    verbose=True)

print("\nConverged:", results["converged"])
print("Iterations:", results["niter"])
print("Log-likelihood: ", results["ll"])
print("Coefficients: ", results["beta"])
print("Standard errors: ", results["se"])

estcoefs_bhhh = results["beta"]
estcoefs_se_bhhh = results["se"]

print('')
print('Parameter estimates and standard errors given by the BHHH optimization routine: ')
print(f"mpg:    {estcoefs_bhhh[0]:.4f} ({estcoefs_se_bhhh[0]:.4f})")
print(f"weight: {estcoefs_bhhh[1]:.4f} ({estcoefs_se_bhhh[1]:.4f})")
print(f"cons:   {estcoefs_bhhh[2]:.4f} ({estcoefs_se_bhhh[2]:.4f})")
print('')
print('-----------------------------------------------------------------------')
print('')