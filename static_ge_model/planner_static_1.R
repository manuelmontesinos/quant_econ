#===============================================================================
# PROGRAM:   Social planner solution to the static general equilibrium model
# AUTHOR:    Manuel V. Montesinos
# REFERENCE: Fehr, H., and Kindermann F. (2018), "Introduction to Computational 
#            Economics using Fortran", Oxford University Press
# DATE:      December 7th, 2019           
#===============================================================================

# Clean the workspace
rm(list=ls())

# Set the value of the parameters
Kbar <- 10
Lbar <- 20
alpha <- 0.3
beta <- c(0.3, 0.6)

# Cobb-Douglas utility function
utility <- function(input) -((input[2]^beta[1]*input[1]^(1-beta[1]))^alpha
                             *((Lbar-input[2])^beta[2]
                             *(Kbar-input[1])^(1-beta[2]))^(1-alpha))
    
# Initial guess for optimization
x <- c(5,5)

# Minimization routine
sol <- optim(x,utility)

# Solution: capital, labor and output
K <- c(sol$par[1],Kbar-sol$par[1])
L <- c(sol$par[2],Lbar-sol$par[2])
Y <- L^beta*K^(1-beta)

# Results
print("GOODS MARKET 1: ")
paste("X1 =",Y[1]," Y1 =",Y[1])
print("GOODS MARKET 2: ")
paste("X2 =",Y[2]," Y2 =",Y[2])
print("LABOR MARKET: ")
paste("L1 =",L[1]," L2 =",L[2]," L =",Lbar)
print("CAPITAL MARKET: ")
paste("K1 =",K[1]," K2 =",K[2]," K =",Kbar)