# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 06:31:40 2020

@author: jlamon02
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np
from itertools import groupby
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

import statsmodels.stats.tests.test_influence

# Load data
def load_data():
    years = np.loadtxt('cleaned_data.csv',delimiter=',',skiprows=1,usecols=0)
    Y = np.loadtxt('cleaned_data.csv',delimiter=',',skiprows=1,usecols=12)
    X = np.genfromtxt('cleaned_data.csv',delimiter=',',skip_header=1,usecols=[1,2,3,4,5,6,7,8,9,10,11])
    return years,Y,X

# Clean data
def clean_dats(years,Y,X,column='any',fill_years = False):
    if column == 'any':
        years_hold = years[~np.isnan(X).any(axis=1)]
        Y_hold = Y[~np.isnan(X).any(axis=1)]
        X_hold = X[~np.isnan(X).any(axis=1)]
    else:
        years_hold = years[~np.isnan(X[:,column]).any(axis=1)]
        Y_hold = Y[~np.isnan(X[:,column]).any(axis=1)]
        X_hold = X[~np.isnan(X[:,column]).any(axis=1)]
    if not fill_years:
        for i in range(1968,1972):
#            print(np.where(years_hold==i))
#            print(np.where(years_hold==i)[0])
            index = np.where(years_hold==i)[0][0]
            Y_hold=np.delete(Y_hold,index,axis=0)
            X_hold=np.delete(X_hold,index,axis=0)
            years_hold=np.delete(years_hold,index,axis=0)
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
def fit_logistic(X_hold,Y_hold):
    res = GLM(Y_hold, X_hold, family=families.Binomial()).fit()#XXX Confirm this with logistic using older XXXX
    return res

# Iterate over columns
def iterate_logistic(X_hold,Y_hold, fixed_columns = [0]):
    l = np.size(fixed_columns)+1
    k = np.shape(X_hold)[1]

    betas = np.zeros([k,l])
    pvalues = np.zeros([k,l])
    aic = np.zeros([k,1])
    bic = np.zeros([k,1])
    
    # Fit constant
    res = fit_logistic(X_hold[:,0],Y_hold)
    betas[0,:] = res.params
    pvalues[0,:] = res.pvalues
    aic[0] = res.aic
    bic[0] = res.bic
    
    NAN = ~np.isnan(X_hold).any(axis=0)
    for i in range(1,k):
        if NAN[i]:
            if i not in fixed_columns:
                columns = fixed_columns.copy()
                columns.append(i)
                res = fit_logistic(X_hold[:,columns],Y_hold)
                betas[i,:] = res.params
                pvalues[i,:] = res.pvalues
                aic[i] = res.aic
                bic[i] = res.bic
    return betas, pvalues,aic,bic

#Now do bootstrapping for chosen model#########################################
#First get index
def boot_index(X,Y,M=5000,block_length=5):
    N = np.size(Y)
    nearest_fifth = block_length*round(N/block_length)
    #print(nearest_fifth)
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

def boot_sample_param(X,Y,indicies,res):
    [M,N] = np.shape(indicies)
    K = np.shape(X)[1]
    
    bootstrap_X = np.zeros([N,K,M])
    bootstrap_Y = np.zeros([N,M])
    
    Y = res.predict()
    for i in range(M):
        bootstrap_X[:,:,i]=X
        bootstrap_Y[:,i] = np.random.binomial(1,Y)
#        for j in range(N):
#            bootstrap_X[j,:,i] = X[indicies[i,j],:]
#            bootstrap_Y[j,i] = np.random.binomial(1,Y[indicies[i,j]])
    return bootstrap_X,bootstrap_Y
#Bootstrap fitting
def boot_fit(bootstrap_X,bootstrap_Y,columns):
    M = np.shape(bootstrap_X)[2]#Number of bootstrap samples
    k = np.size(columns)#Number of parameters
    beta_boot = np.zeros([M,k])
    
    for i in range(M):
#        print(i)
#        print(bootstrap_X[:,columns,i])
#        print(np.sum(bootstrap_Y[:,i]))
#        np.save('boot_X',bootstrap_X[:,columns,i])
#        np.save('boot_Y',bootstrap_Y[:,i])
        
        X_hold=normalize(bootstrap_X[:,columns,i])
        X_hold = add_constant(X_hold,bootstrap_Y[:,i])
        res = fit_logistic(X_hold,bootstrap_Y[:,i])
        beta_boot[i,:] = res.params
    return beta_boot

def boot_master(X,Y,columns,M=5000,block_length = 5,param=True,res=[]):
    indicies = boot_index(X,Y,M,block_length)
    if param:
        bootstrap_X,bootstrap_Y = boot_sample_param(X,Y,indicies,res)
    else:
        bootstrap_X,bootstrap_Y = boot_sample(X,Y,indicies,block_length)
    beta_boot = boot_fit(bootstrap_X,bootstrap_Y,columns)
    return beta_boot,bootstrap_X,bootstrap_Y

#Now do simulation of GCMS#####################################################
#Load GCMS--Need to finish for Jared --XX
def load_GCMS(hist_temp,hist_precip):
    Temp_GCM = np.loadtxt('GCM_Temp.csv',delimiter=',',skiprows=63)[:,1:13] #Drop year column load in 1962-2100
    Precip_GCM = np.loadtxt('GCM_Precip.csv',delimiter=',',skiprows=63)[:,1:13] #Drop year column load in 1962-2100
    Years_GCM = np.loadtxt('GCM_Precip.csv',delimiter=',',skiprows=63)[:,0] # load in 1962-2100
    hist_mean = 150.6115385#Historical mean @ GP
    Precip_GCM = convert_precip(Precip_GCM,hist_mean)#Convert to fraction of mean
    Temp_GCM,Precip_GCM = splice_history(Temp_GCM,Precip_GCM,hist_temp,hist_precip)
    return Temp_GCM,Precip_GCM,Years_GCM
    
def convert_precip(precip,hist_mean):
    #Convert precip values to percent of historical average values
    return precip/hist_mean

#Splice in historical data
def splice_history(Temp_GCM,Precip_GCM,hist_temp,hist_precip):
    N = np.size(hist_precip)
    for i in range(12):
        Temp_GCM[0:N,i] = hist_temp
        Precip_GCM[0:N,i] = hist_precip
    return Temp_GCM,Precip_GCM        

def splice_flood(flood,scenario,boot,hist_flood,N_prob):
    N = np.size(hist_flood)
    for i in range(N_prob):
        flood[0:N,scenario,i,boot] = hist_flood
    return flood
        
#Do random Simulation
def simulate_GCM_futures(hist_flood,bootstrap_X,bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=5000,N_prob=1000):
    #Initialize
    N_scen = np.shape(Temp_GCM)[1]#Number of GCM/RCP scenarios
    prob = np.zeros([139,N_scen,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob
    flood = np.zeros([139,N_scen,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    cum_flood = np.zeros([139,N_scen,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    waits = np.zeros([139,N_scen,N_prob,M_boot])#135 years (1962-2100) by 12 GCM/RCP by N_prob by M_boot
    #Loop over boot
    for boot in range(M_boot):
        X_fu = bootstrap_X[:,:,boot]
        Y_fu = bootstrap_Y[:,boot]
        beta_fu = beta_boot[boot,:]
        X_mean = np.mean(X_fu,0)
        X_std = np.std(X_fu,0)
        #Loop over scenario
        for scenario in range(N_scen):
            #Compute Probabilities
            forc = np.zeros([139,3])
            forc[:,0] = 1
            forc[:,1] = (Temp_GCM[:,0]-X_mean[0])/X_std[0]
            forc[:,2] = (Precip_GCM[:,1]-X_mean[1])/X_std[1]
            
            prob[:,scenario,boot] = compute_prob(beta_fu,forc)
            #Simulate Floods
            flood[:,scenario,:,boot] = simulate_ice_jams(prob[:,scenario,boot],N_prob)
            #Splice in Historical Floods
            flood = splice_flood(flood,scenario,boot,hist_flood,N_prob)
            #Count Cum Floods
            [cum_flood[:,scenario,:,boot],cum_quant] = make_cum(flood[:,scenario,:,boot])
            waits[:,scenario,:,boot]=projected_waits(cum_flood[:,scenario,:,boot])
            #Wait Times
        print(boot)
    return prob,flood,cum_flood,waits

def compute_prob(boot_fu,forc):
    prod = np.matmul(forc,boot_fu)
    return np.exp(prod)/(1+np.exp(prod))

def simulate_ice_jams(prob,N_prob):
    #L is the length of the simulation (i.e. 2020-2100=>80)
    #N_prob is the number of replicates
    L = len(prob)
    X = np.zeros([L,N_prob])
    for i in range(L):
        X[i,:] = np.random.rand(N_prob)<prob[i]
    return X

def make_cum(X):
    L=np.shape(X)[0]
    X=np.cumsum(X,axis=0)
    cum_quant = np.zeros([L,5])
    for i in range(L): #compute 1%,25%,50%,75%,99%
        cum_quant[i,:] = np.percentile(X[i,:],[1,25,50,75,99])
    return X,cum_quant

def projected_waits(cum_flood):
    L,N = np.shape(cum_flood)
    #waits = np.zeros([L,N])
    X = np.ones([L,N])
    for i in range(L-1,1,-1):
        X[i-1,cum_flood[i,:] == cum_flood[i-1,:]] = X[i,cum_flood[i,:] == cum_flood[i-1,:]]+1
    
    X[0,:] = X[1,:]+1
    X[L-1,:] = np.inf
    for i in range(L-2,0,-1):
        X[i,(X[i,:]!=1).__and__(X[i+1,:]==np.inf)]=np.inf
    return X

def mimic_bootstrap(results,X,Y,M_boot=10):
    boot_betas=np.random.multivariate_normal(results.params,results.cov_params(),size=[M_boot])
    columns = [1,3]
    [N,k] = np.shape(X[:,columns])
    bootstrap_X = np.zeros([N,k,M_boot])
    bootstrap_Y = np.zeros([N,M_boot])
    for i in range(M_boot):
        bootstrap_X[:,:,i] = X[:,columns]
        bootstrap_Y[:,i] = Y
    return boot_betas,bootstrap_X,bootstrap_Y
    
#Count wait times
#Compution expected wait times
#Compute median wait times
#GCM master