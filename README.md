# Quantitative Economics
A collection of computational methods for conducting research in economics and finance. Files included:

## Optimization Algorithms
- Genetic algorithm:
  - [genetic_algorithm_example.py](genetic_algorithm/genetic_algorithm_example.py) (Python).

## Discrete Choice Models
- Estimation of a binary logit model: 
  - [BLogit_Car.do](discrete_choice/binary_logit/BLogit_Car.do) (Stata).
  - [BLogit_Car_PyStata.py](discrete_choice/binary_logit/BLogit_Car_PyStata.py) (PyStata).
- Estimation of a binary probit model: 
  - [BProbit_Car.do](discrete_choice/binary_probit/BProbit_Car.do) (Stata).
  - [BProbit_Car.m](discrete_choice/binary_probit/BProbit_Car.m) (Matlab).
  - [BProbit_Car.py](discrete_choice/binary_probit/BProbit_Car.py) (Python, using several gradient-free optimization algorithms).
  - [BProbit_Car_Powell.py](discrete_choice/binary_probit/BProbit_Car_Powell.py).
  - For optimization based on the BFGS algorithm, use this Matlab function to compute the log-likelihood: [bprobit_llike.m](discrete_choice/binary_probit/bprobit_llike.m).
  - For optimization based on the Newton-Raphson algorithm, use this Matlab function: [bprobit_nr.m](discrete_choice/binary_probit/bprobit_nr.m).
- Estimation of a multinomial logit model: 
  - [mlogit_insurance.do](discrete_choice/multinomial_logit/mlogit_insurance.do) (Stata).
  - [mlogit_insurance.m](discrete_choice/multinomial_logit/mlogit_insurance.m) (Matlab).
  - For optimization based on the BFGS algorithm, use this Matlab function to compute the log-likelihood: [mlogit_insurance_llike.m](discrete_choice/multinomial_logit/mlogit_insurance_llike.m).
  - For optimization based on the Newton-Raphson algorithm, use this Matlab function: [mlogit_nr.m](discrete_choice/multinomial_logit/mlogit_nr.m).

## The Static General Equilibrium Model
- Social planner solution to the static general equilibrium model: 
  - [planner_static_1.R](static_ge_model/planner_static_1.R) (R).
  - [planner_static_1.py](static_ge_model/planner_static_1.py) (Python).
  - [planner_static_1.ipynb](static_ge_model/planner_static_1.ipynb) (Jupyter Notebook).
- Market solution to the static general equilibrium model: 
  - [market_static_1.R](static_ge_model/market_static_1.R) (R).
  - [market_static_1.py](static_ge_model/market_static_1.py) (Python).
  - [market_static_1.ipynb](static_ge_model/market_static_1.ipynb) (Jupyter Notebook).
- Market solution to the static general equilibrium model with variable labor supply: 
  - [market_static_labor.py](static_ge_model/market_static_labor.py) (Python).
  - [market_static_labor.ipynb](static_ge_model/market_static_labor.ipynb) (Jupyter Notebook).
- Static general equilibrium model with government activity: 
  - [static_ge_government.py](static_ge_model/static_ge_government.py) (Python).
  - [static_ge_government.ipynb](static_ge_model/static_ge_government.ipynb) (Jupyter Notebook).

## Introduction to Dynamic Programming
- All-in-one solution to the cake-eating problem: 
  - [cake_eating_all_in_one.ipynb](intro_dynamic_programming/cake_eating_all_in_one.ipynb) (Jupyter Notebook).
  - [prog08_01.jl](intro_dynamic_programming/prog08_01.jl) (Julia).
- Analytical solution to the cake-eating problem: 
  - [cake_eating_analytic.ipynb](intro_dynamic_programming/cake_eating_analytic.ipynb) (Jupyter Notebook).
  - [prog08_02.jl](intro_dynamic_programming/prog08_02.jl) (Julia).  

## References
- [Fehr, H., and Kindermann, F. (2018): "Introduction to Computational Economics using Fortran", Oxford University Press.](https://www.ce-fortran.com/)
- [Fletcher, R. (1987): "Practical Methods of Optimization", John Wiley & Sons.](https://archive.org/details/practicalmethods0000flet)
