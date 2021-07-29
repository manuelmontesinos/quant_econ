function [llike] = bprobit_llike(yobs, xobs, b0)
%==========================================================================
% FUNCTION: Log-likelihood of the binary probit model
% 
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% INPUT:
%   yobs <-- (nobs-by-1) vector of observations of the dependent variable,
%            i.e., indicator of discrete choice (0,1).
%   xobs <-- (nobs-by-k) matrix of explanatory variables.
%   betas <- (k-by-1) vector of parameters to be estimated.
%
% OUTPUT:
%   llike <- (scalar) value of the log-likelihood.
%==========================================================================

% Number of explanatory variables
k = size(xobs, 2);

% Compute conditional choice probabilities
xb = xobs*b0;
pxb = normcdf(xb);

% Compute the log-likelihood
llike = yobs.*log(pxb) + (1-yobs).*log(1-pxb);
llike = sum(llike);
llike = -llike;

%==========================================================================
end