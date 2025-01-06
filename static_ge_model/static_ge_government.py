#===============================================================================
# PROGRAM:   The static general equilibrium model with government activity
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
print("STATIC GENERAL EQUILIBRIUM MODEL WITH GOVERNMENT ACTIVITY")
print(" ")

# Capital endowment
Kbar = 10

# Time endowment
Tbar = 30

# Parameters of Cobb-Douglas preferences and technology
alpha = np.array([0.3, 0.4])
beta = np.array([0.3, 0.6])

# Level of the public good
G = 3

# Wage income tax
tauw = 0

# Interest income tax
taur = 0

# Consumption tax
tauc = np.array([0, 0])

# Function to determine market equilibrium
def markets(x):

    # Copy producer prices and taxes
    q = np.array([1, x[0]])
    w = x[1]
    r = x[2]

    # Set tax rates (uncomment respective lines for different results)
    taur = x[3]
    #tauw = -x[3]
    #tauc = np.array([x[3], 0])
    #tauc = np.array([0, x[3]])

    # Calculate consumer prices and total income
    p = q*(1 + tauc)
    wn = w*(1 - tauw)
    rn = r*(1 - taur)
    Ybarn = wn*Tbar + rn*Kbar

    # Get market equations
    ms1 = alpha[0]*Ybarn/p[0] + G - (beta[0]/w)**beta[0] * \
        ((1-beta[0])/r)**(1-beta[0]) * q[0]*(alpha[0]*Ybarn/p[0]+G)
    ms2 = 1/p[1] - (beta[1]/w)**beta[1] * \
            ((1-beta[1])/r)**(1-beta[1])*q[1]/p[1]
    ms3 = beta[0]/w * q[0]*(alpha[0]*Ybarn/p[0]+G) + \
            beta[1]/w * q[1]*alpha[1]*Ybarn/p[1] + \
            (1-alpha[0]-alpha[1])*Ybarn/wn - Tbar
    ms4 = q[0]*G - tauc[0]/(1+tauc[0])*alpha[0]*Ybarn - \
            tauc[1]/(1+tauc[1])*alpha[1]*Ybarn - \
            tauw*w*(Tbar-(1-alpha[0]-alpha[1])/wn * Ybarn) - taur*r*Kbar
    return [ms1, ms2, ms3, ms4]

# Initial guess for prices and tax rates in equilibrium
x0 = [0.5, 0.5, 0.5, 0.5]

# Find market equilibrium
equil = fsolve(markets, x0)
q = [1, equil[0]]
w = equil[1]
r = equil[2]

# Set tax rates (uncomment the line of the tax to endogeneize)
taur = equil[3]
#tauw = equil[3]
#tauc = np.array([equil[3], 0])
#tauc = np.array([0, equil[3]])

# Calculate consumer prices and total income
p = q*(1+tauc)
wn = w*(1-tauw)
rn = r*(1-taur)

# Calculate other economic variables
Ybarn = wn*Tbar + rn*Kbar
Xd = alpha*Ybarn/p
Y = [Xd[0]+G, Xd[1]]
ell = (1-alpha[0]-alpha[1])*Ybarn/wn
L = beta*q*Y/w
K = (1-beta)*q*Y/r
U = Xd[0]**alpha[0] * Xd[1]**alpha[1] * ell**(1-alpha[0]-alpha[1])

# Results
print("GOODS MARKET 1: ")
print("X1 =", Xd[0], " G =", G, " Y1 =", Y[0])
print("q1 =", q[0], " p1 =", p[0], " tc1=", tauc[0])
print(" ")
print("GOODS MARKET 2: ")
print("X2 =", Xd[1], " G =", 0, " Y2 =", Y[1])
print("q2 =", q[1], " p2 =", p[1], " tc2=", tauc[1])
print(" ")
print("LABOR MARKET: ")
print("L1 =", L[0], " L2 =", L[1], " T-F =", Tbar-ell)
print("w =", w, " wn =", wn, " tw =", tauw)
print(" ")
print("CAPITAL MARKET: ")
print("K1 =", K[0], " K2 =", K[1], " K =", Kbar)
print(" r =", r, " rn =", rn, " tr =", taur)
print(" ")
print("GOVERNMENT: ")
print("tc1 =", tauc[0]*q[0]*Xd[0], " tc2 =", tauc[1]*q[1]*Xd[1], \
      " tw =", tauw*w*(Tbar-ell), " tr =", taur*r*Kbar, " G =", q[0]*G)
print(" ")
print("UTILITY: ")
print("U =", U)
print(" ")

print("-----------------------------------------------------------------------")
