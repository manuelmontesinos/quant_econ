#===============================================================================
# PROGRAM: Example of implementation of the genetic algorithm for optimization.
#
# AUTHOR: Manuel V. Montesinos (ROCKWOOL Foundation Berlin).
#
# THIS VERSION: September 2024.
#
# DESCRIPTION: 
#===============================================================================

# Import modules
import numpy as np
import pandas as pd
import pygad
from make_fitness import make_fitness

print('')
print('EXAMPLE OF IMPLEMENTATION OF THE GENETIC ALGORITHM FOR OPTIMIZATION')
print('')

#-------------------------------------------------------------------------------

# Set seed
np.random.seed(13)

# Import the data
auto = pd.read_csv('../discrete_choice/binary_probit/auto.csv')

# Organize the data and add a constant
choice = auto['foreign'].to_numpy()
mpg = auto['mpg'].to_numpy()
weight = auto['weight'].to_numpy()
constant = np.ones(len(choice))
regressors = np.column_stack((mpg, weight, constant))

#-------------------------------------------------------------------------------

# Number of parameters to be estimated
k = regressors.shape[1]

# Prepare PyGAD inputs
fitness_function = make_fitness(choice, regressors)

# Function to print the best solution in each generation
def on_generation(ga):
    best_sol, best_fit, _ = ga.best_solution()
    if ga.generations_completed % 10 == 0:
        print(f"Gen {ga.generations_completed:4d}  best_loglike≈{best_fit: .6f}")

# Number of generations
num_generations = 200

# Number of solutions to be selected as parents (cannot be greater than number of
# solutions in the population (4))
num_parents_mating = 4

# Number of genes in the solution
num_genes = k

# Define the gene space to limit the range of each gene (parameter)
gene_space = [{"low": -5.0, "high": 5.0}] * k

# Number of solutions in the population
sol_per_pop = 30

# Parent selection type 
parent_selection_type = "sss"

# Number of parents to keep in the next generation
keep_parents = 1

# Crossover type
crossover_type = "single_point"

# Mutation type
mutation_type = "random"

# Percentage of genes to mutate
mutation_percent_genes = 50

# Create an instance of pygad.GA
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       sol_per_pop=sol_per_pop,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=on_generation)

# Run the genetic algorithm
ga_instance.run()

# Retrieve the best solution
best_params, best_fitness, solution_idx = ga_instance.best_solution()
print("Best params:", best_params)
print("Best log-likelihood (≈ fitness):", best_fitness)
print("Best NLL:", -best_fitness)

# Solution obtained
# Best params: [ 2.41247088 -0.02486435  3.37247157]
# Best log-likelihood (≈ fitness): -234.3797255256226
# Best NLL: 234.3797255256226