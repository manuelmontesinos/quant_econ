%==========================================================================
% PROGRAM: Estimation of a multionomial logit model of insurance choice
%
% AUTHOR: Manuel V. Montesinos (Universitat Autonoma de Barcelona and
% Barcelona GSE)
%
% THIS VERSION: July 2021
%
% DESCRIPTION: This program estimates the parameters of a multinomial logit
% model of insurance choice, using health insurance data from
% https://www.stata-press.com/data/r17/sysdsn1.dta
%
%==========================================================================

% Close, clear all and set seed
close all;
clear;
clc;
rng(13);

%% Start

disp('')
disp('ESTIMATION OF A MULTINOMIAL LOGIT MODEL OF INSURANCE CHOICE')
disp('')

%% Set paths

    % Programs folder
    programs = '/Users/montesinos/Documents/GitHub/quant_econ/maximum_likelihood_estimation/multinomial_logit';
    userpath(programs)

    % Data folder
    cleandata = '/Users/montesinos/Documents/GitHub/quant_econ/maximum_likelihood_estimation/multinomial_logit';
    addpath(cleandata)

%% Import the data

    % Set up the import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 5);

    % Specify range and delimiter
    opts.DataLines = [2, Inf];
    opts.Delimiter = ",";

    % Specify column names and types
    opts.VariableNames = ["patid", "age", "male", "nonwhite", "insure"];
    opts.VariableTypes = ["double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Import the data (the output is in table format)
    sysdsn1mod = readtable(fullfile(cleandata, "sysdsn1_mod.csv"), opts);

    % Convert the table into matrix
    data = sysdsn1mod{:,:};

    % Organize the data and add a constant
    choice = data(:, 5);
    regressors = [data(:, 2:4), ones(size(data,1), 1)];

%% Estimate the model using 'fminunc' (Quasi-Newton, BFGS method)

    % Names of the parameters
    namesparam = {'age_2','male_2','nonwhite_2','cons_2',...
                  'age_3','male_3','nonwhite_3','cons_3'};

    % Initial values of the parameter vector
    b0 = zeros(size(namesparam,2),1);

    % Estimate the model
    options1 = optimset('Display','iter','PlotFcn',@optimplotfval,...
        'Diagnostics','on','MaxIter',1e20,'MaxFunEvals',1e10,...
        'TolX',1e-20,'TolFun',1e-20,'GradObj','off');

    disp('====> Estimate the parameters')
    disp('')

    [bhat1,fv1,ef1,o1,grad1,hess1] = ...
        fminunc(@(bhat1) mlogit_insurance_llike(choice, regressors, bhat1),...
        b0, options1);

    se1 = sqrt(diag(inv(hess1)));

%% Estimate the model using a function that implements Newton-Raphson

    % Names of the parameters
    namesparam = {'age_2','male_2','nonwhite_2','cons_2',...
                  'age_3','male_3','nonwhite_3','cons_3'};

    % Initial values of the parameter vector
    b0 = zeros(size(namesparam, 2),1);

    % Estimate the model
    disp('====> Estimate the parameters')
    disp('')

    [bhat2, se2, fval2] = mlogit_nr(choice, regressors, b0);
