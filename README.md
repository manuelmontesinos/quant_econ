# Quantitative Economics
A collection of computational methods for conducting research in economics and finance. Files included:

## Discrete Choice Models
- Estimation of a multinomial logit model of insurance choice: [mlogit_insurance.do](https://github.com/manuelmontesinos/quant_econ/blob/main/discrete_choice/multinomial_logit/mlogit_insurance.do) (Stata), [mlogit_insurance.m](https://github.com/manuelmontesinos/quant_econ/blob/main/discrete_choice/multinomial_logit/mlogit_insurance.m) (Matlab).
  - For optimization based on the BFGS algorithm, use this Matlab function to compute the log-likelihood: [mlogit_insurance_llike.m](https://github.com/manuelmontesinos/quant_econ/blob/main/discrete_choice/multinomial_logit/mlogit_insurance_llike.m).
  - For optimization based on the Newton-Raphson algorithm, use this Matlab function: [mlogit_nr.m](https://github.com/manuelmontesinos/quant_econ/blob/main/discrete_choice/multinomial_logit/mlogit_nr.m).

## The Static General Equilibrium Model
- Social planner solution to the static general equilibrium model: [planner_static_1.R](https://github.com/montesinosmv/quant_econ/blob/master/static_ge_model/planner_static_1.R) (R), [planner_static_1.py](https://github.com/manuelmontesinos/quant_econ/blob/main/static_ge_model/planner_static_1.py) (Python).
- Market solution to the static general equilibrium model: [market_static_1.R](https://github.com/montesinosmv/quant_econ/blob/master/static_ge_model/market_static_1.R) (R), [market_static_1.py](https://github.com/manuelmontesinos/quant_econ/blob/main/static_ge_model/market_static_1.py) (Python).
- Market solution to the static general equilibrium model with variable labor supply: [market_static_labor.py](https://github.com/manuelmontesinos/quant_econ/blob/main/static_ge_model/market_static_labor.py) (Python).
- Static general equilibrium model with government activity: [static_ge_government.py](https://github.com/manuelmontesinos/quant_econ/blob/main/static_ge_model/static_ge_government.py) (Python).

## Introduction to Dynamic Programming
- All-in-one solution to the cake-eating problem: [cake_eating_all_in_one.ipynb](https://github.com/manuelmontesinos/quant_econ/blob/master/intro_dynamic_programming/cake_eating_all_in_one.ipynb) (Jupyter Notebook), [prog08_01.jl](https://github.com/montesinosmv/quant_econ/blob/master/intro_dynamic_programming/prog08_01.jl) (Julia).
- Analytical solution to the cake-eating problem: [cake_eating_analytic.ipynb](https://github.com/manuelmontesinos/quant_econ/blob/master/intro_dynamic_programming/cake_eating_analytic.ipynb) (Jupyter Notebook), [prog08_02.jl](https://github.com/montesinosmv/quant_econ/blob/master/intro_dynamic_programming/prog08_02.jl) (Julia).

## References
- [Fehr, H., and Kindermann, F. (2018): "Introduction to Computational Economics using Fortran", Oxford University Press.](https://www.ce-fortran.com/)
- [Fletcher, R. (1987): "Practical Methods of Optimization", John Wiley & Sons.](https://archive.org/details/practicalmethods0000flet)
