#===============================================================================
# PROGRAM:   Market solution to the static general equilibrium model
# AUTHOR:    Manuel V. Montesinos
# REFERENCE: Fehr, H., and Kindermann F. (2018), "Introduction to Computational 
#            Economics using Fortran", Oxford University Press
# DATE:      December 24th, 2019           
#===============================================================================

# Call libraries
library(pracma)

# Clean the workspace
rm(list=ls())

# Capital endowment
Kbar <- 10

# Labor endowment
Lbar <- 20

# Parameters of Cobb-Douglas preferences and technologies
alpha <- 0.3
beta <- c(0.3,0.6)

# Function to determine market equilibrium
markets <- function(x0) {
    ## copy prices (the price of good 1 is normalized)
    p <- c(1,x0[1])
    w <- x0[2]
    r <- x0[3]
    ## compute total income
    Ybar <- w*Lbar + r*Kbar
    ## market equations
    ms1 <- 1/p[1] - (beta[1]/w)^beta[1]*((1-beta[1])/r)^(1-beta[1])
    ms2 <- 1/p[2] - (beta[2]/w)^beta[2]*((1-beta[2])/r)^(1-beta[2])
    ms3 <- beta[1]*alpha*Ybar/w + beta[2]*(1-alpha)*Ybar/w - Lbar
    markets_sol <- c(ms1,ms2,ms3)
}

# Initial guess for the price vector
x <- c(0.5,0.5,0.5)

# Find the root of the market equations using Broyden's method
equil <- broyden(markets,x)

# Copy equilibrium prices 
market_equil <- unname(unlist(equil[1]))

p <- c(1,market_equil[1])
w <- market_equil[2]
r <- market_equil[3]

# Calculate other economic variables in equilibrium
Ybar <- w*Lbar + r*Kbar
Y <- c(alpha*Ybar/p[1],(1-alpha)*Ybar/p[2])
L <- beta*p*Y/w
K <- (1-beta)*p*Y/r
U <- Y[1]^alpha*Y[2]^(1-alpha)

# Results: same quantities as in the planner's solution
print("GOODS MARKET 1: ")
paste("X1 =",Y[1]," Y1 =",Y[1]," q1 =",p[1]," p1 =",p[1])
print("GOODS MARKET 2: ")
paste("X2 =",Y[2]," Y2 =",Y[2]," q2 =",p[2]," p2 =",p[2])
print("LABOR MARKET: ")
paste("L1 =",L[1]," L2 =",L[2]," L =",Lbar," w =",w)
print("CAPITAL MARKET: ")
paste("K1 =",K[1]," K2 =",K[2]," K =",Kbar," r =",r)
print("UTILITY: ")
paste("U =",U)

#===============================================================================