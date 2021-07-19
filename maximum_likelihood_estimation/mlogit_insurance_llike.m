function [llike] = mlogit_insurance_llike(yobs, xobs, b0)
%==========================================================================
% FUNCTION: Log-likelihood of the multinomial logit model
% 
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% INPUT:
%   yobs <-- (nobs-by-1) vector of observations of the dependent variable,
%            i.e., indicator of discrete choice (1, 2, ..., nalt)
%   xobs <-- (nobs-by-k) matrix of explanatory variables
%   betas <- (k*(nalt-1)-by-1) vector of parameters to be estimated
%
% OUTPUT:
%   llike <- (scalar) value of the log-likelihood
%==========================================================================

% Number of explanatory variables
k = size(xobs, 2);

% Number of choice alternatives
nalt = max(yobs);

% Compute conditional choice probabilities
b0 = [zeros(k, 1) reshape(b0, k, nalt-1)];
xb = xobs*b0;
sumexpe = sum(exp(xb'), 1);
pxb = exp(xb)./transpose(sumexpe);

% Compute the log-likelihood
llike = 0;
for jj = 1:nalt
    llike = llike + sum((yobs==jj).*log(pxb(:,jj)),1);
end
llike = -llike;

%==========================================================================
end