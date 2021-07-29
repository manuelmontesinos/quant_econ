%==========================================================================
% PROGRAM: Estimation of a binary probit model
%
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% DESCRIPTION: This program estimates the parameters of a binary probit
% model explaining whether a car is foreign based on its weight and
% mileage, using data from http://www.stata-press.com/data/r13/auto. The
% model is Pr(foreign = 1) = Phi(beta_0 + beta_1*weight + beta_2*mpg).
%==========================================================================

% Close, clear all and set seed
close all;
clear;
clc;
rng(13);

%% Start

disp('')
disp('ESTIMATION OF A BINARY PROBIT MODEL')
disp('')

%% Set paths

    % Programs folder
    programs = '/Users/montesinos/Documents/GitHub/quant_econ/discrete_choice/binary_probit';
    userpath(programs)

    % Data folder
    cleandata = '/Users/montesinos/Documents/GitHub/quant_econ/discrete_choice/binary_probit';
    addpath(cleandata)
    
%% Import the data 

    % Set up the import options and import the data
    opts = delimitedTextImportOptions("NumVariables", 3);

    % Specify range and delimiter
    opts.DataLines = [2, Inf];
    opts.Delimiter = ",";

    % Specify column names and types
    opts.VariableNames = ["mpg", "weight", "foreign"];
    opts.VariableTypes = ["double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    
    % Import the data (the output is in table format)
    auto = readtable(fullfile(cleandata, "auto.csv"), opts);
    
    % Convert the table into a matrix
    data = auto{:,:};
    
    % Organize the data and add a constant
    choice = data(:,3);
    regressors = [data(:,1:2), ones(size(data,1),1)];
    
%% Estimate the model using 'fminunc' (Quasi-Newton, BFGS method)

    % Names of the parameters
    namesparam = {'mpg','weight','_cons'};
    
    % Initial values of the parameter vector
    b0 = zeros(size(namesparam,2), 1);
    
    % Optimization options
    options1 = optimset('Display','iter','MaxIter',1e20,'MaxFunEvals',1e10,...
        'TolX',1e-20,'TolFun',1e-20,'GradObj','off');

    % Estimate the model
    disp('====> Estimate the parameters')
    disp('')
    
    [bhat1,fval1,~,~,grad1,hess1] = ...
        fminunc(@(bhat1) bprobit_llike(choice, regressors, bhat1),...
        b0, options1);
    
    % Standard errors
    phixb1 = normpdf(regressors*bhat1);
    pxb1 = normcdf(regressors*bhat1);
    Avarb1 = (((-phixb1./(1-pxb1)).*(phixb1./pxb1)).*regressors)'*regressors;
    Avarb1 = inv(-Avarb1);
    se1 = sqrt(diag(Avarb1));
    
%% Estimate the model using a function that implements Newton-Raphson

    % Names of the parameters
    namesparam = {'mpg','weight','_cons'};

    % Initial values of the parameter vector
    b0 = zeros(size(namesparam,2), 1);
    
    % Estimate the model 
    disp('====> Estimate the parameters')
    disp('')
    
    [bhat2, se2, fval2] = bprobit_nr(choice, regressors, b0);