#==============================================================================#
# PROGRAM: The dynamic solution to the cake-eating problem
# AUTHOR: Manuel V. Montesinos
# DATE: March 2nd, 2020
# REFERENCE: Fehr, H. and Kindermann, F. (2018): "Computational
#            Economics using Fortran", Oxford University Press
#==============================================================================#

# Call libraries
using Plots

# Set model parameters
γ = 0.5
egam = 1-1/γ
β = 0.95
a0 = 100

# Other variables
TT = 200
NA = 1000
t = 1:TT

# Initialize vectors
c_t = zeros(TT)
a_t = zeros(TT)
a = zeros(NA)
c = zeros(NA)
V = zeros(NA)

# Calculate the time path of consumption
a_t[1] = a0
c_t[1] = a_t[1]*(1-β^γ)

for it in 2:TT
    a_t[it] = a_t[it-1]-c_t[it-1]
    c_t[it] = a_t[it]*(1-β^γ)
end

# Plot the time path of consumption
ctplot = plot(t,c_t
    ,title=""
    ,lw=3, xlabel="Time", ylabel="Consumption",label="")

# Plot policy and value function
for ia in 1:NA
    a[ia] = 1+ia/NA*a0
    c[ia] = a[ia]*(1-β^γ)
    V[ia] = (1-β^γ)^(-1/γ)*a[ia]^egam/egam
end

# Plot policy function
pplot = plot(a,c
    ,title=""
    ,lw=3, lc=:green, xlabel="Current level of resources a"
    ,ylabel="Policy function c(a)",label="")

# Plot value function
vplot = plot(a,V
    ,title=""
    ,lw=3, lc=:red, xlabel="Current level of resources a"
    ,ylabel="Value function V(a)",label="")

outplot = plot(ctplot, pplot, vplot, layout=3)
