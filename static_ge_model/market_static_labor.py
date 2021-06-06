#===============================================================================
# PROGRAM:   Market solution to the static general equilibrium model with
#            variable labor supply
# AUTHOR:    Manuel V. Montesinos
# REFERENCE: Fehr, H., and Kindermann F. (2018), "Introduction to Computational
#            Economics using Fortran", Oxford University Press
# DATE:      June 6th, 2021
#===============================================================================

# Import libraries
import numpy as np
from scipy.optimize import fsolve

print("-----------------------------------------------------------------------")
print(" ")
print("MARKET SOLUTION TO THE STATIC GENERAL EQUILIBRIUM MODEL WITH VARIABLE")
print("LABOR SUPPLY")
print(" ")

# Capital endowment
Kbar = 10

# Time endowment of the household
Tbar = 30

# Parameters of Cobb-Douglas preferences and technologies
alpha = np.array([0.3, 0.4])
beta = np.array([0.3, 0.6])

# Function to determine market equilibrium
def markets(x):
    # Copy prices (the price of good 1 is normalized)
    p = [1, x[0]]
    w = x[1]
    r = x[2]
    # Compute total income
    Ybar = w*Tbar + r*Kbar
    # Market equations
    ms1 = 1/p[0] - (beta[0]/w)**beta[0] * ((1-beta[0])/r)**(1-beta[0])
    ms2 = 1/p[1] - (beta[1]/w)**beta[1] * ((1-beta[1])/r)**(1-beta[1])
    ms3 = beta[0]*alpha[0]*Ybar/w + beta[1]*alpha[1]*Ybar/w + \
        (1-alpha[0]-alpha[1])*Ybar/w - Tbar
    return [ms1, ms2, ms3]

# Initial guess for the price vector
x0 = [0.5, 0.5, 0.5]

# Find the root of the market equations
equil = fsolve(markets, x0)
p = [1, equil[0]]
w = equil[1]
r = equil[2]

# Calculate other variables in equilibrium
Ybar = w*Tbar + r*Kbar
Y = [alpha[0]*Ybar/p[0], alpha[1]*Ybar/p[1]]
ell = (1-alpha[0]-alpha[1])*Ybar/w
L = beta*p*Y/w
K = (1-beta)*p*Y/r
U = Y[0]**alpha[0] * Y[1]**alpha[1] * ell**(1-alpha[0]-alpha[1])

# Results
print("GOODS MARKET: ")
print("X1 =", Y[0], " X2 =", Y[1], " p1 =", p[0], " p2 =", p[1])
print(" ")
print("LABOR MARKET: ")
print("L1 =", L[0], " L2 =", L[1], " T-F =", Tbar-ell, " w =", w)
print(" ")
print("CAPITAL MARKET: ")
print("K1 =", K[0], " K2 =", K[1], " K =", Kbar, " r =", r)
print(" ")
print("UTILITY: ")
print("U =", U)
print(" ")

print("-----------------------------------------------------------------------")
