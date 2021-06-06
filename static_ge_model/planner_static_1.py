#===============================================================================
# PROGRAM:   Social planner solution to the static general equilibrium model
# AUTHOR:    Manuel V. Montesinos
# REFERENCE: Fehr, H., and Kindermann F. (2018), "Introduction to Computational
#            Economics using Fortran", Oxford University Press
# DATE:      June 5th, 2021
#===============================================================================

# Import libraries
import numpy as np
from scipy.optimize import minimize

print("-----------------------------------------------------------------------")
print(" ")
print("SOCIAL PLANNER SOLUTION TO THE STATIC GENERAL EQUILIBRIUM MODEL")
print(" ")

# Set the value of the parameters
Kbar = 10
Lbar = 20
alpha = 0.3
beta = np.array([0.3, 0.6])

# Cobb-Douglas utility function
def utility(x):
    return -(x[1]**beta[0] * x[0]**(1-beta[0]))**alpha * \
        ((Lbar-x[1])**beta[1] * (Kbar-x[0])**(1-beta[1]))**(1-alpha)

# Initial guess for optimization
x0 = np.array([5, 5])

# Minimization routine
sol = minimize(utility, x0, method='nelder-mead', \
                options={'xatol': 1e-8, 'disp': True})
print(" ")

# Solution: capital, labor and output
K = np.array([sol.x[0], Kbar-sol.x[0]])
L = np.array([sol.x[1], Lbar-sol.x[1]])
Y = L**beta * K**(1-beta)

# Results
print("GOODS MARKET: ")
print("X1 =", Y[0], " X2 =", Y[1])
print(" ")
print("LABOR MARKET: ")
print("L1 =",L[0], " L2 =", L[1], " L =", Lbar)
print(" ")
print("CAPITAL MARKET: ")
print("K1 =", K[0], " K2 =", K[1], " K =", Kbar)
print(" ")
print("-----------------------------------------------------------------------")
