# Import modules
import numpy as np
from bprobit_nll import bprobit_nll

def make_fitness(yobs, xobs):
    
    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness function for PyGAD to evaluate candidate solutions based on the negative log-likelihood
        of a binary probit model.

        Parameters:
        solution (list or np.ndarray): Candidate solution (parameter vector) to evaluate.
        solution_idx (int): Index of the candidate solution in the population.

        Returns:
        float: Fitness value (non-negative) for the candidate solution.
        """ 
        # Converts the candidate solution to a NumPy array for numerical operations
        betas = np.asarray(solution)

        # Calculates the negative log-likelihood for the binary probit model using the parameters
        nll = bprobit_nll(betas, yobs, xobs)

        # PyGAD prefers non-negative fitness. Shift if needed.
        fitness = -nll
        
        # Ensures the returned fitness is non-negative, as required by PyGAD
        return fitness
    
    return fitness_func