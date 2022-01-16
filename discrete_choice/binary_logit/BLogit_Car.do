/*=============================================================================*
  PROJECT: Estimation of a binary logit model.

  AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and Barcelona 
  School of Economics).
 
  THIS VERSION: January 2022.
 
  DESCRIPTION: This do-file estimates the parameters of a binary logit model 
  explaining whether a car is foreign based on its weight and mileage, using
  data from http://www.stata-press.com/data/r13/auto.
*=============================================================================*/

	******* Set path

	cd "write path"

	******* Open log-file

	cap log close blogit_car
	log using "BLogit_Car.smcl", replace name(blogit_car)

	di " "
	di " "
	di "	====> ESTIMATION OF A BINARY LOGIT MODEL:", as text
	di " "

	******* Load the data

	use "http://www.stata-press.com/data/r13/auto", clear
	
	/****** Estimate the model and predict the CCPs: 
			Pr(foreign = 1) = exp(beta_0 + beta_1*weight 
			+ beta_2*mpg)/(1+exp(beta_0 + beta_1*weight + beta_2*mpg)) */
	
	logit foreign mpg weight
	predict phat, pr
	
	******* Save the dataset in csv format

	keep foreign weight mpg 
	label val foreign .
	export delimited using "auto.csv", quote replace	
	
	******* Close log-file

	cap log close blogit_car

*==============================================================================*
	
