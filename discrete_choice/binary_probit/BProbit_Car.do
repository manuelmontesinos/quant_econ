/*=============================================================================*
  PROJECT: Estimation of a binary probit model

  AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and Barcelona 
  GSE)
 
  THIS VERSION: July 2021
 
  DESCRIPTION: This do-file estimates the parameters of a binary probit model 
  explaining whether a car is foreign based on its weight and mileage, using
  data from http://www.stata-press.com/data/r13/auto
*=============================================================================*/

	******* Set path

	cd "../quant_econ/discrete_choice/binary_probit"

	******* Open log-file

	cap log close bprobit_car
	log using "BProbit_Car.smcl", replace name(bprobit_car)

	di " "
	di " "
	di "	====> ESTIMATION OF A BINARY PROBIT MODEL:", as text
	di " "

	******* Load the data

	use "http://www.stata-press.com/data/r13/auto", clear
	
	/****** Estimate the model: 
			Pr(foreign = 1) = Phi(beta_0 + beta_1*weight + beta_2*mpg) */
	
	probit foreign mpg weight
	
	******* Save the dataset in csv format

	keep foreign weight mpg 
	label val foreign .
	export delimited using "auto.csv", quote replace
	
	******* Close log-file

	cap log close bprobit_car

*==============================================================================*
	
