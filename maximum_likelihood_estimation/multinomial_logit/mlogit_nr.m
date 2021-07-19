function [best, sebest, llik] = mlogit_nr(yobs, xobs, b0)
%==========================================================================
% FUNCTION: Estimate the a multinomial logit model by maximum likelihood,
% minimizing the function by Newton-Raphson
% 
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% INPUT: 
%   yobs <- (nosb-by-1) vector of observations of the dependent variable,
%           i.e., indicator of discrete choice (1, 2, ..., nalt)
%   xobs <- (nobs-by-k) matrix of explanatory variables
%   b0   <- (k*(nalt-1)-by-1) vector with initial guess for parameter
%           values
%
% OUTPUT:
%   best   <- (k*(nalt-1)-by-1) vector of parameter estimates
%   sebest <- Standard errors 
%   llik   <- Value of the log-likelihood in the maximum
%==========================================================================

% Number of observations and number of explanatory variables
kpar = size(xobs, 2);

% Number of discrete alternatives
nalt = max(yobs);

% Optimization parameters
eps = 1e-6;
iter = 1;
criter = 1000;

% Start optimization
while criter > eps
    
    % Compute conditional choice probabilities
    b0 = [zeros(kpar, 1) reshape(b0, kpar, nalt-1)];
    xb = xobs*b0;
    sumexpe = sum(exp(xb'), 1);
    pxb = exp(xb)./transpose(sumexpe);
    
    % Compute the log-likelihood
    myzero = 1e-16;
    llik = 0;

    jj = 1;
    while jj <= nalt
      llik = llik+sum((yobs==jj).*log(pxb(:,jj)+myzero), 1);
      jj = jj+1;
    end
    
    % Compute the gradient and the hessian
    d1like = zeros(kpar*(nalt-1), 1);
    d2like = zeros(kpar*(nalt-1), kpar*(nalt-1));

    jj = 2;
    while jj <= nalt
      indj1 = (jj-2)*kpar+1;
      indj2 = (jj-1)*kpar;
      d1like(indj1:indj2) = sum(xobs.*((yobs==jj)-pxb(:,jj)), 1);
      k = 2;
      while k <= nalt
         indk1 = (k-2)*kpar+1;
         indk2 = (k-1)*kpar;
         if jj ~= k
            d2like(indj1:indj2,indk1:indk2) = (xobs.*pxb(:,jj))'*(xobs.*pxb(:,k));
         else
             d2like(indj1:indj2,indk1:indk2) = ...
                 (xobs.*pxb(:,jj))'*(xobs.*pxb(:,k))-(xobs.*pxb(:,jj))'*xobs;
         end
         k = k+1;
      end
      jj = jj+1;
    end

    % Update criterion, parameter values and iteration 
    b0 = reshape(b0(:, 2:end), kpar*(nalt-1), 1);
    b1 = b0 - inv(d2like)*d1like;
    criter = sqrt((b1-b0)'*(b1-b0));
    b0 = b1;
    iter = iter+1;
end

% Parameter estimates
best = b0;

% Standard errors
Avarb = inv(-d2like);
sebest = sqrt(diag(Avarb));

%==========================================================================
end