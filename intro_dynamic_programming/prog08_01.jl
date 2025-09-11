#==============================================================================#
# PROGRAM: All-in-one solution to the cake-eating problem
# AUTHOR: Manuel V. Montesinos
# DATE: February 26th, 2020
# REFERENCE: Fehr, H. and Kindermann, F. (2018): "Computational
#            Economics using Fortran", Oxford University Press
#==============================================================================#

# Call libraries
using Plots

# Set model parameters
γ = 0.5
β = 0.95
a0 = 100

# Other variables
TT = 200
c_t = zeros(TT)
t = 1:TT

# Calculate the time path of consumption

for it in 1:TT
    c_t[it] = β^(it*γ)*(1-β^γ)*a0
end

# Plot the result
plot(t,c_t
    ,title="Optimal consumption path in the cake-eating problem"
    ,lw=3, xlabel="Time", ylabel="Consumption",label="")
