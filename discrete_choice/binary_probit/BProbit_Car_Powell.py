#===============================================================================
# PROGRAM: Estimation of a binary probit model using the Powell algorithm
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
from bprobit_llike import bprobit_llike
from scipy import stats
from scipy import optimize
from scipy.optimize import minimize

# Set seed
np.random.seed(13)

# Start
print('')
print('ESTIMATION OF A BINARY PROBIT MODEL USING THE POWELL ALGORITHM')
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

#-------------------------------------------------------------------------------

# Initial values of the parameter vector
b0 = np.zeros(3)

# Dictionary to store the history of function evaluations
history = {'Nfeval': 0, 'nfeval': [], 'fval': [], 'params': []}

# Estimate the model using the Powell algorithm in SciPy (gradient-free). It 
# performs quite well and takes 434 function evaluations to converge (some more
# than Nelder-Mead)
print('{0:4s}   {1:9s}'.format('Iter', 'f(X)'))
print('Estimate the model using the Powell algorithm in SciPy: ')
outmin = minimize(bprobit_llike, b0, args=(choice, regressors, history), \
    method='Powell', options={'disp': True})

# Plot the objective function value against the number of function evaluations
plt.figure(figsize=(10, 6))
plt.plot(history['nfeval'], history['fval'], linestyle='-', color='b')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Value vs. Number of Function Evaluations')
plt.grid(True)
plt.show()

# Plot the parameter values against the number of function evaluations
params = np.array(history['params'])
param_names = ['mpg', 'weight', 'constant']
plt.figure(figsize=(10, 6))
for i, name in enumerate(param_names):
    plt.plot(history['nfeval'], params[:, i], linestyle='-', label=name)
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Parameter Values')
plt.title('Parameter Values vs. Number of Function Evaluations')
plt.legend()
plt.grid(True)
plt.show()

# Print the parameter estimates
print('')
print('Parameter estimates using SciPy Powell algorithm: ')
print('mpg:    ', outmin.x[0])
print('weight: ', outmin.x[1])
print('cons:   ', outmin.x[2])
print('')
print('-----------------------------------------------------------------------')
print('')