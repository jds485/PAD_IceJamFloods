# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 06:31:40 2020

@author: jlamon02
"""
import sys
import warnings
import math
from scipy import stats
from scipy.stats import chi2
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from itertools import groupby
import statsmodels
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

import statsmodels.stats.tests.test_influence

import os
owd = os.getcwd()
os.chdir('C:\\Users\\js4yd\\Documents\\LogisticPADIJF\\FirthRegression')
from firth_regression import *
os.chdir(owd)
del owd

# Load data
def load_data():
    years = np.loadtxt('cleaned_data.csv',delimiter=',',skiprows=1,usecols=0)
    Y = np.loadtxt('cleaned_data.csv',delimiter=',',skiprows=1,usecols=12)
    X = np.genfromtxt('cleaned_data.csv',delimiter=',',skip_header=1,usecols=[1,2,3,4,5,6,7,8,9,10,11])
    return years,Y,X

# Clean data
def clean_dats(years,Y,X,column='any'):
    if column == 'any':
        years_hold = years[~np.isnan(X).any(axis=1)]
        Y_hold = Y[~np.isnan(X).any(axis=1)]
        X_hold = X[~np.isnan(X).any(axis=1)]
    else:
        years_hold = years[~np.isnan(X[:,column]).any(axis=1)]
        Y_hold = Y[~np.isnan(X[:,column]).any(axis=1)]
        X_hold = X[~np.isnan(X[:,column]).any(axis=1)]
    return years_hold,Y_hold,X_hold

# Normalize independent variable
def normalize(X):
    k = np.shape(X)[1]
    for i in range(k):
        X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
    return X

# Add constant
def add_constant(X,Y):
    X = np.concatenate((np.ones([np.size(Y),1]),X),axis=1)
    return X

# Fit logisitic regression
def fit_logistic(X_hold,Y_hold,Firth=False):
    if not Firth:
        res = GLM(Y_hold, X_hold, family=families.Binomial()).fit()#XXX Confirm this with logistic using older XXXX
        # AICc adjustment
        res.aicc = statsmodels.tools.eval_measures.aicc(res.llf, nobs=res.nobs, df_modelwc=res.df_model+1)
        # Correct BIC
        res.bic = statsmodels.tools.eval_measures.bic(res.llf, nobs=res.nobs, df_modelwc=res.df_model+1)
    else:
        #Do Firth's logistic regression
        (rint, rbeta, rbse, rfitll) = fit_firth(Y_hold, X_hold, start_vec = None)
        # Wald test
        waldp = 2. * (1. - stats.norm.cdf(abs(rbeta[0]/rbse[0])))
        
        # LRT
        null_X = np.delete(arr=X_hold,obj=range(int(np.size(X_hold)/len(X_hold)))[1:int(np.size(X_hold)/len(X_hold))],axis=1)
        (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(Y_hold, null_X, start_vec = None)
        lrstat = -2.*(null_fitll - rfitll)
        lrt_pvalue = 1.
        if lrstat > 0.: # non-convergence
            lrt_pvalue = stats.chi2.sf(lrstat, 1)
        
        #Using this as a way to return a model in the same class as GLM.
        res = GLM(Y_hold, X_hold, family=families.Binomial()).fit()
        # AICc adjustment
        res.aicc_GLM = statsmodels.tools.eval_measures.aicc(res.llf, nobs=res.nobs, df_modelwc=res.df_model+1)
        
        # AICc adjustment for Firth model
        aicc = statsmodels.tools.eval_measures.aicc(rfitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        # AIC
        aic = statsmodels.tools.eval_measures.aic(rfitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        # BIC
        bic = statsmodels.tools.eval_measures.bic(rfitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        #Store parameters, standard errors, likelihoods, and statistics
        rint = np.array([rint])
        rbeta = np.array(rbeta)
        res.params = np.concatenate([rint,rbeta])
        res.bse = rbse
        res.llf = rfitll
        res.llnull = null_fitll
        res.aicc = aicc
        res.aic = aic
        res.bic = bic
        res.waldp = waldp
        res.lrstat = lrstat
        res.lrt_pval = lrt_pvalue
        
        #Get Wald p vals for parameters
        res.pvalues = 1. - chi2.cdf(x=(res.params/res.bse)**2, df=1)
        
    return res

# Iterate over columns
def iterate_logistic(X_hold,Y_hold, fixed_columns = [0], Firth=False):
    l = np.size(fixed_columns)+1
    k = np.shape(X_hold)[1]

    betas = np.zeros([k,l])
    pvalues = np.zeros([k,l])
    aic = np.zeros([k,1])
    aicc = np.zeros([k,1])
    bic = np.zeros([k,1])
    
    # Fit constant
    if Firth:
        null_X = np.delete(arr=X_hold,obj=range(int(np.size(X_hold)/len(X_hold)))[1:int(np.size(X_hold)/len(X_hold))],axis=1)
        (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(Y_hold, null_X, start_vec = None)
        
        #Using this as a way to return a model in the same class as GLM.
        res = GLM(Y_hold, null_X, family=families.Binomial()).fit()
        # AICc adjustment for Firth model
        res.aicc = statsmodels.tools.eval_measures.aicc(null_fitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        # AIC
        res.aic = statsmodels.tools.eval_measures.aic(null_fitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        # BIC
        res.bic = statsmodels.tools.eval_measures.bic(null_fitll, nobs=res.nobs, df_modelwc=res.df_model+1)
        #Store parameters, standard errors, likelihoods, and statistics
        res.params = np.array([null_intercept])
        #Get Wald p vals for parameters
        res.pvalues = 1. - chi2.cdf(x=(res.params/null_bse)**2, df=1)
    else:
        res = fit_logistic(X_hold[:,0],Y_hold)
    
    betas[0,:] = res.params
    pvalues[0,:] = res.pvalues
    aic[0] = res.aic
    aicc[0] = res.aicc
    bic[0] = res.bic
    
    NAN = ~np.isnan(X_hold).any(axis=0)
    for i in range(1,k):
        if NAN[i]:
            if i not in fixed_columns:
                columns = fixed_columns.copy()
                columns.append(i)
                res = fit_logistic(X_hold[:,columns],Y_hold, Firth=Firth)
                betas[i,:] = res.params
                pvalues[i,:] = res.pvalues
                aic[i] = res.aic
                aicc[i] = res.aicc
                bic[i] = res.bic
    return betas, pvalues,aic,aicc,bic

#Now do bootstrapping for chosen model#########################################
#First get index
def boot_index(X,Y,M=5000,block_length=5):
    N = np.size(Y)
    nearest_fifth = block_length*round(N/block_length)
    print(nearest_fifth)
    index = range(N)
    indicies = np.zeros([M,nearest_fifth])
    for i in range(0,nearest_fifth,block_length):
        #print(i)
        index = np.random.randint(0,N,size=[M])
        indicies[:,i] = index
        for j in range(0,block_length):
            indicies[:,i+j] = indicies[:,i]+j
            for k in range(block_length):
                indicies[indicies[:,i+j]==N+k,i+j] = k #wrap around
    indicies=indicies.astype('int')
    return indicies

#Next do bootstrap sampling
def boot_sample(X,Y,indicies,block_length = 5):
    [M,N] = np.shape(indicies)
    K = np.shape(X)[1]
    
    bootstrap_X = np.zeros([N,K,M])
    bootstrap_Y = np.zeros([N,M])
    
    for i in range(M):
        bootstrap_X[:,:,i]=X[indicies[i,:],:]
        bootstrap_Y[:,i] = Y[indicies[i,:]]
        for j in range(N):
            bootstrap_X[j,:,i] = X[indicies[i,j],:]
            bootstrap_Y[j,i] = Y[indicies[i,j]]
    return bootstrap_X,bootstrap_Y

#Bootstrap fitting
def boot_fit(bootstrap_X,bootstrap_Y,columns, Firth=False):
    M = np.shape(bootstrap_X)[2]#Number of bootstrap samples
    k = np.size(columns)#Number of parameters
    beta_boot = np.zeros([M,k])
    
    for i in range(M):
        print(i)
#        print(bootstrap_X[:,columns,i])
#        print(bootstrap_Y[:,i])
#        np.save('boot_X',bootstrap_X[:,columns,i])
#        np.save('boot_Y',bootstrap_Y[:,i])
        res = fit_logistic(bootstrap_X[:,columns,i],bootstrap_Y[:,i], Firth=Firth)
        beta_boot[i,:] = res.params
    return beta_boot

def boot_master(X,Y,columns,M=5000,block_length = 5):
    indicies = boot_index(X,Y,M,block_length)
    bootstrap_X,bootstrap_Y = boot_sample(X,Y,indicies,block_length)
    beta_boot = boot_fit(bootstrap_X,bootstrap_Y,columns)
    return beta_boot,bootstrap_X,bootstrap_Y

#Now do simulation of GCMS#####################################################
#Load GCMS--Need to finish for Jared --XX
def load_GCMS(hist_FC_DDF,hist_GP_Precip):
    hist_mean = 150.650000#Historical mean @ GP
    hist_FC_freeze_mom = [np.mean(hist_FC_DDF),np.std(hist_FC_DDF)] #moments of host DDF @ FC
    hist_GP_mom = [np.mean(hist_GP_Precip),np.std(hist_GP_Precip)]
    Temp = np.loadtxt('GCM_Temp.csv',delimiter=',',skiprows=47)[:,1:13] #Drop year column, load in 2020-2100
    Precip = np.loadtxt('GCM_Precip.csv',delimiter=',',skiprows=47)[:,1:13] #Drop year column, load in 2020-2100
    Precip = convert_precip(Precip,hist_mean)
    Temp,Precip=splice_history(Temp,Precip,A)

def convert_precip(precip,hist_mean):
    #Convert precip values to percent of historical average values
    return precip/hist_mean

#Splice in historical data

#Do random Simulation
def simulate_GCM_futures(beta_boot,X,bootstrarp_X,bootstrap_Y,M_boot=5000,N_prob=1000):
    Temp,Precip=splice_history(Temp,Precip,A)#Circle_back
    prob = np.zeros([135,12,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob
    flood = np.zeros([135,12,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    cum_flood = np.zeros([135,12,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    waits = np.zeros([135,12,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    
    for boot in range(M_boot):
        X_fu = bootstrap_X[:,:,boot]
        Y_fu = bootstrap_Y[:,boot]
        X_mean = np.mean(X_fu,0)
        X_std = np.std(X_fu,0)
        
        for scenario in range(12):
            #Merge history w/ GCM and normalize
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            prob[:,scenario,boot] = compute_prob(beta_boot[boot],X_fu)
            flood[:,scenario,boot,:] = simulate_ice_jams(prob[:,scenario,boot],N_prob)
            #Splice historical_floods
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            [cum_flood[:,scenario,boot,:],cum_quant] = make_cum(flood[:,scenario,boot,:])#make cumulative floods, Need to load historical data
            waits[:,scenario,boot,:]=projected_waits(cum_flood[:,scenario,boot,:])

def compute_prob(beta,X_fu):
    N = np.shape(X_fu)[1]
    prob = np.exp(beta[0]+beta[1]*X_fu[:,0]+beta[2]*X_fu[2])/(np.exp(beta[0]+beta[1]*X_fu[:,0]+beta[2]*X_fu[2])+1)
    return prob

def simulate_ice_jams(prob,N):
    #L is the length of the simulation (i.e. 2020-2100=>80)
    #N is the number of replicates
    L = len(prob)
    X = np.zeros([L,N])
    for i in range(L):
        X[i,:] = np.random.rand(N)<prob[i]
    return X
#Count wait times
#Compution expected wait times
#Compute median wait times
#GCM master