# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:58:00 2020

@author: jlamon02
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = False)

X = normalize(X)

X = add_constant(X,Y)

#Test constant model with GLM vs. logistic regression
res_GLM = fit_logistic(X[:,[0,1,4]],Y)

import statsmodels.api as sm
logit=sm.Logit(Y,X[:,[0,1,4]])
res_SM = logit.fit()

#Jared: Betas and aic seem the same.  P-values are the same out a ways.  bic are different.
# I belived the SM bic.  Everything else looks consistent between the two.
#Looking at seperation
plt.figure()
plt.scatter(X[Y==0,2],Y[Y==0])
plt.scatter(X[Y==1,2],Y[Y==1])
plt.xlabel('Fort Verm DDF')

plt.figure()
plt.scatter(X[Y==0,4],Y[Y==0])
plt.scatter(X[Y==1,4],Y[Y==1])
plt.xlabel('BL Precip')

plt.figure()
plt.scatter(X[Y==0,4],X[Y==0,2])
plt.scatter(X[Y==1,4],X[Y==1,2])
plt.xlabel('BL Precip')
plt.ylabel('Fort Chip Verm')

# Example of iterative model building
[betas, pvalues,aic,bic]=iterate_logistic(X,Y, fixed_columns = [0])
# Three parameter models
[betas3, pvalues3,aic3,bic3]=iterate_logistic(X,Y, fixed_columns = [0,4])
# Cross parameter models
cross=np.reshape(X[:,1]*X[:,4],(-1,1))
X_hold = np.concatenate((X[:,[0,1,4]],cross),axis=1)
res_cross = fit_logistic(X_hold,Y)
#Best model is BL-GP & Fort Vermillion DDF.

[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = True)
#Now bootstrap#################################################################
#[beta_boot,bootstrap_X,bootstrap_Y] = boot_master(X,Y,columns=[0,1,4],M=5000,block_length = 5,param=True,res=res_GLM)

#Now GCM#######################################################################
boot_betas,bootstrap_X,bootstrap_Y=mimic_bootstrap(res_GLM,X,Y,M_boot=50)
Temp_GCM,Precip_GCM,Years_GCM = load_GCMS(X[:,1],X[:,3])
prob,flood,cum_flood,waits = simulate_GCM_futures(Y,bootstrap_X,bootstrap_Y,boot_betas,Temp_GCM,Precip_GCM,M_boot=50,N_prob=1000)

#Now plots#####################################################################
from utils_figures import *
import matplotlib.pyplot as plt
GCMs = ['RCP85 - HadGEM2-ES','RCP85 - ACCESS1-0','RCP85 - CanESM2','RCP85 - CCSM4','RCP85 - CNRM-CM5','RCP85 - MPI-ESM-LR','RCP45 - HadGEM2-ES','RCP45 - ACCESS1-0','RCP45 - CanESM2','RCP45 - CCSM4','RCP45 - CNRM-CM5','RCP45 - MPI-ESM-LR']
#Make double plots
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:120]=moving_average(plt_perc[i,:],20)
        plt_perc2[i,0:120]=moving_average(plt_perc2[i,:],20)
        for j in range(120,139):
            plt_perc[i,j] = np.mean(plt_perc[i,j:139])
            plt_perc2[i,j] = np.mean(plt_perc2[i,j:139])
    percentile_fill_plot_double(plt_perc,plt_perc2,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.001,0.5],Names=['RCP85','RCP45'])
#All prob
plt_perc = np.zeros([12,139])
for scenario in range(12):
    
    #Probability
    plt_perc[scenario,:] = np.percentile(prob[:,scenario,:],50,axis=1)

    plt_perc[scenario,0:120]=moving_average(plt_perc[scenario,:],20)
    for j in range(120,139):
        plt_perc[scenario,j] = np.mean(plt_perc[scenario,j:139])

percentile_plot_single(plt_perc,title='Ice Jam Flood Projection',ylabel='IJF Probability',scale='log',ylim=[0.001,0.5],split='gcm',Names=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR'])

#Plot GCM Data
#Temp RCPs
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
plt.figure()
for i in range(6):
    plt.plot(range(1950,2096),moving_average(Temp_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1950,2096),moving_average(Temp_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Fort Chip DDF')

#Prec RCPs
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
plt.figure()
for i in range(6):
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Beaverlodge Precip')

#Prec GCMs
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
plt.figure()
for i in range(2):
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i],10),'b',linewidth=3,label=GCMs[0])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i+1],10),'g',linewidth=3,label=GCMs[1])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i+2],10),'r',linewidth=3,label=GCMs[2])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i+3],10),'c',linewidth=3,label=GCMs[3])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i+4],10),'m',linewidth=3,label=GCMs[4])
    plt.plot(range(1950,2096),moving_average(Precip_GCM[:,6*i+5],10),'y',linewidth=3,label=GCMs[5])
    #plt.plot(range(1950,2096),moving_average(Precip[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(GCMs)
plt.xlabel('Year')
plt.ylabel('Beaverlodge Precip')

#Survival Analysis#############################################################