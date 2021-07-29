function [bhat, se, llike] = bprobit_nr(yobs, xobs, b0)
%==========================================================================
% FUNCTION: Estimate a binary probit model using the Newton-Raphson
% algorithm for optimization
% 
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% INPUT:
%   yobs <-- (nobs-by-1) vector of observations of the dependent variable,
%            i.e., indicator of discrete choice (0, 1).
%   xobs <-- (nobs-by-k) matrix of explanatory variables.
%   b0 <---- (k-by-1) vector of initial values for the parameters.
%
% OUTPUT:
%   bhat <- (k-by-1) vector of estimated parameters
%   se <--- (k-by-1) vector of standard errors.
%   llike <- (scalar) value of the log-likelihood in the maximum.
%==========================================================================

% Number of observations
nobs = size(yobs, 1);

% Number of explanatory variables
k = size(xobs, 2);

% Optimization parameters
iter = 1;
llike = 1000;
criter1 = 1000;
criter2 = 1000;
eps1 = 1e-6;
eps2 = 1e-6;

while (criter1 > eps1) || (criter2 > eps2)
    
    % Compute conditional choice probabilities
    xb = xobs*b0;
    pxb = normcdf(xb);
    
    % Compute the log-likelihood
    llike = yobs.*log(pxb) + (1-yobs).*log(1-pxb);
    
    % Compute the gradient
    phixb = normpdf(xb);
    lambda = yobs.*(phixb./pxb) + (1-yobs).*(-phixb./(1-pxb));
    dlogLb = xobs'*lambda;
    
    % Compute the hessian
    d2logLb = ((lambda.*(lambda+xobs*b0)).*xobs)'*xobs;
    
    % Update parameter values, criterion and iteration
    b1 = b0 + inv(d2logLb)*dlogLb;
    criter1 = sqrt((b1-b0)'*(b1-b0));
    criter2 = sqrt(dlogLb'*dlogLb);
    b0 = b1;
    iter = iter+1;
    
end

% Parameter estimates 
bhat = b0;

% Standard errors
lambda1 = -phixb./(1-pxb);
lambda2 = phixb./pxb;
Avarb = ((lambda1.*lambda2).*xobs)'*xobs;
Avarb = inv(-Avarb);
se = sqrt(diag(Avarb));

% Value of the log-likelihood in the maximum
llike = -sum(llike);

%==========================================================================
end