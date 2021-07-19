*==========================================================================*
* PROJECT: Estimation of a multinomial logit model of insurance choice
*
* AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and 
* Barcelona GSE)
*
* THIS VERSION: July 2021
*
* DESCRIPTION: This do-file estimates the parameters of a multinomial logit
* model of insurance choice, using health insurance data from 
* https://www.stata-press.com/data/r17/sysdsn1.dta
*==========================================================================*
	
	******* Set path
	
	cd "/Users/montesinos/Documents/GitHub/quant_econ/maximum_likelihood_estimation"
	
	******* Open log-file 
	
	cap log close mlogit_insurance
	log using "mlogit_insurance.smcl", replace name(mlogit_insurance)

	di " "
	di " "
	di "	====> ESTIMATION OF A MULTINOMIAL LOGIT MODEL OF INSURANCE CHOICE:", as text 
	di " "	

	******* Load the data 
	
	use "https://www.stata-press.com/data/r17/sysdsn1.dta", clear
	
	******* Estimate the model
	
	mlogit insure nonwhite male age, baseoutcome(1)
	
	******* Save the dataset in csv format 
	
	keep patid insure nonwhite male age 
	keep if !missing(patid, insure, nonwhite, male, age)
	label val insure .
	export delimited using "sysdsn1_mod.csv", quote replace 
	
	******* Close log-file
	
	cap log close mlogit_insurance
	
*===========================================================================
