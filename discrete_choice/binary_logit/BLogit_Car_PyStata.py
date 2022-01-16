'''
================================================================================
PROGRAM: Estimation of a binary logit model using PyStata.

AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and Barcelona
School of Economics).

THIS VERSION: January 2022.

DESCRIPTION: This program estimates the parameters of a binary logit model
explaining whether a car is foreign, based on its weight and mileage, using data
from http://www.stata-press.com/data/r13/auto. The logit model is estimated from
Stata.
================================================================================
'''

# Set-up Stata. 'config' takes two arguments: the directory where Stata is
# installed in your computer (type 'display c(sysdir_stata)' in Stata to find
# it) and the edition of Stata you have (be/se/mp))
import stata_setup
stata_setup.config('/Applications/Stata/', 'se')
from pystata import stata

# Import other modules
import pandas as pd

#-------------------------------------------------------------------------------

# Import the data into a dataframe
auto = pd.read_csv('auto.csv')

# Load data to Stata
stata.pdataframe_to_data(auto, force=True)

# Summarize the data
stata.run('summarize')

# Estimate the logit model
stata.run('logit foreign mpg weight')

# Load e-returned results into Python and print the coefficient estimates
eresult = stata.get_ereturn()
betahat = eresult['e(b)'][0]
print('Parameter estimates: ')
print('mpg:      ', betahat[0])
print('weight:   ', betahat[1])
print('constant: ', betahat[2])

#-------------------------------------------------------------------------------

print('')
