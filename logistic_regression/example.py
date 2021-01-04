# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:58:00 2020

@author: jlamon02
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import seaborn as sb
import pandas as pd
# set seed
random.seed(123020)
np.random.seed(123020)

# Load data
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = False)

X = normalize(X)

X = add_constant(X,Y)

#Test constant model with GLM vs. logistic regression
res_GLM = fit_logistic(X[:,[0,2,4]],Y)

resBase = copy.deepcopy(res_GLM)
#Firth regression method
import statsmodels.api as sm
res_Firth = fit_logistic(X[:,[0,2,4]],Y,Firth=True, resBase = resBase)
del resBase

logit=sm.Logit(Y,X[:,[0,2,4]])
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
plt.xlabel('BL/GP Nov.-Apr. Precip.')

plt.figure()
plt.scatter(X[Y==0,2],X[Y==0,4])
plt.scatter(X[Y==1,2],X[Y==1,4])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Fort Verm. DDF')

plt.figure()
plt.scatter(X[Y==0,1],X[Y==0,4])
plt.scatter(X[Y==1,1],X[Y==1,4])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Fort Chip. DDF')

plt.figure()
plt.scatter(X[Y==0,1],X[Y==0,6])
plt.scatter(X[Y==1,1],X[Y==1,6])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Freeze-up Stage (Beltaos, 2014)')

plt.figure()
plt.scatter(X[Y==0,1],X[Y==0,7])
plt.scatter(X[Y==1,1],X[Y==1,7])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Melt Test')

# Example of iterative model building
[betas, pvalues,aic,aicc,bic]=iterate_logistic(X,Y, fixed_columns = [0])
[betasf, pvaluesf,aicf,aiccf,bicf]=iterate_logistic(X,Y, fixed_columns = [0],Firth=True)
# Three parameter models
[betas3, pvalues3,aic3,aicc3,bic3]=iterate_logistic(X,Y, fixed_columns = [0,4])
[betas3f, pvalues3f,aic3f,aicc3f,bic3f]=iterate_logistic(X,Y, fixed_columns = [0,4],Firth=True)
#Freeze
[betas3frf, pvalues3frf,aic3frf,aicc3frf,bic3frf]=iterate_logistic(X,Y, fixed_columns = [0,6],Firth=True)
# Cross parameter models
#cross=np.reshape(X[:,1]*X[:,4],(-1,1))
#X_hold = np.concatenate((X[:,[0,1,4]],cross),axis=1)
#res_cross = fit_logistic(X_hold,Y)
#res_cross_Firth = fit_logistic(X_hold,Y,Firth=True)

cross=np.reshape(X[:,2]*X[:,4],(-1,1))
X_hold = np.concatenate((X[:,[0,2,4]],cross),axis=1)
res_cross = fit_logistic(X_hold,Y)
resBase = copy.deepcopy(res_cross)
res_cross_Firth = fit_logistic(X_hold,Y,Firth=True,resBase=resBase)
del resBase
#Note: interaction term is not significant with profile likelihood p-value = .134
#Best model is BL-GP & Fort Vermillion DDF.
#p-values for best model from profile likelihood in R:
#Int: 1.252043e-11     B1: 1.532832e-02     B2: 1.969853e-03

#Constant and Interaction Only
X_hold = np.concatenate((X[:,[0]],cross),axis=1)
resBase = copy.deepcopy(res_cross)
res_crossOnly_Firth = fit_logistic(X_hold,Y,Firth=True,resBase=resBase)
del resBase

# Four parameter models
[betas4f, pvalues4f,aic4f,aicc4f,bic4f]=iterate_logistic(X,Y, fixed_columns = [0,2,4],Firth=True)
#Freeze
[betas4frf, pvalues4frf,aic4frf,aicc4frf,bic4frf]=iterate_logistic(X,Y, fixed_columns = [0,4,6],Firth=True)
# Five parameter models
[betas5f, pvalues5f,aic5f,aicc5f,bic5f]=iterate_logistic(X,Y, fixed_columns = [0,2,4,8],Firth=True)
#Freeze
[betas5frf, pvalues5frf,aic5frf,aicc5frf,bic5frf]=iterate_logistic(X,Y, fixed_columns = [0,2,4,6],Firth=True)

#Now bootstrap#################################################################
resBase = copy.deepcopy(res_GLM)
#reload data for use in boot_master function
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = False)
[beta_boot,bootstrap_X,bootstrap_Y] = boot_master(X,Y,columns=[1,3],M=1000,param=True,res=res_Firth, Firth=True, resBase=resBase)
#Parametric bootstrap using MVN dist.
#boot_betas,bootstrap_X,bootstrap_Y=mimic_bootstrap(res_GLM,X,Y,M_boot=50)

#pair plot of the beta empirical distribution
betas_boot_df = pd.DataFrame(data=beta_boot, columns = ['Constant', 'DDF Fort Verm.', 'BL/GP Precip.'])
sb.set_style('darkgrid')
test = sb.pairplot(betas_boot_df, diag_kind = 'kde', plot_kws={"s":13})

#Now GCM#######################################################################
#Load the dam filling years
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = True)

Temp_GCM,Precip_GCM,Years_GCM = load_GCMS(X[:,1],X[:,3])
prob,flood,cum_flood,waits = simulate_GCM_futures(Y,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=1000,N_prob=1000)
#For use with MVN dist. parametric bootstrap
#prob,flood,cum_flood,waits = simulate_GCM_futures(Y,bootstrap_X,bootstrap_Y,boot_betas,Temp_GCM,Precip_GCM,M_boot=5000,N_prob=1000)

#Now plots#####################################################################
from utils_figures import *
import matplotlib.pyplot as plt
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
window = 20
#Make double plots
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='linear',ylim=[0.00001,0.2],Names=RCPs,window=window)
#Median
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='linear',ylim=[0.00001,0.2],Names=RCPs,window=window)
#Log scale
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00001,0.5],Names=RCPs,window=window)
#Median
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00001,0.5],Names=RCPs,window=window)

#Average in log space
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(np.log10(plt_perc[i,:]),window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(np.log10(plt_perc2[i,:]),window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='linear',ylim=[-6,0],Names=RCPs,window=window)
#Median in log space
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(np.log10(plt_perc[i,:]),window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(np.log10(plt_perc2[i,:]),window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='linear',ylim=[-6,0],Names=RCPs,window=window)

#Return period
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(1/plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],1/plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Return Period',scale='log',ylim=[1,10000],Names=RCPs,window=window)
#Median
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=RunningMedian(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(1/plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],1/plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Return Period',scale='log',ylim=[1,10000],Names=RCPs,window=window)

#All prob
plt_perc = np.zeros([np.shape(Temp_GCM)[1],np.shape(Temp_GCM)[0]])
for scenario in range(np.shape(Temp_GCM)[1]):
    
    #Probability
    plt_perc[scenario,:] = np.percentile(prob[:,scenario,:],50,axis=1)

    plt_perc[scenario,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[scenario,:],window)
    #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
    #    plt_perc[scenario,j] = np.mean(plt_perc[scenario,j:np.shape(Temp_GCM)[0]])

percentile_plot_single(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title='Ice Jam Flood Projection',ylabel='IJF Probability',scale='log',ylim=[0.00001,0.5],split='gcm',Names=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR'],window=window)

#Plot GCM Data
#Temp RCPs
plt.figure()
for i in range(6):
    plt.plot(range(1962,2091),moving_average(Temp_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1962,2091),moving_average(Temp_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Fort Verm DDF')

#Prec RCPs
plt.figure()
for i in range(6):
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Beaverlodge Precip')

#Prec GCMs
plt.figure()
for i in range(2):
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i],10),'b',linewidth=3,label=GCMs[0])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i+1],10),'g',linewidth=3,label=GCMs[1])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i+2],10),'r',linewidth=3,label=GCMs[2])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i+3],10),'c',linewidth=3,label=GCMs[3])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i+4],10),'m',linewidth=3,label=GCMs[4])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6*i+5],10),'y',linewidth=3,label=GCMs[5])
    #plt.plot(range(1950,2096),moving_average(Precip[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(GCMs)
plt.xlabel('Year')
plt.ylabel('Beaverlodge Precip')

#Survival Analysis#############################################################
median = survival(waits)
A = np.zeros([np.shape(Temp_GCM)[0]-10+1,np.shape(Temp_GCM)[1]])
for i in range(np.shape(Temp_GCM)[1]):
    A[:,i] = moving_average(median[:,i],n=10)

plt.figure()
for i in range(6):
    plt.plot(Years_GCM[38:89],median[38:89,i],'g',linewidth = 5)
    plt.plot(Years_GCM[38:89],median[38:89,i+6],'b',linewidth = 5)

plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#to 2100 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(Years_GCM,median[:,i],'g',linewidth = 5)
    plt.plot(Years_GCM,median[:,i+6],'b',linewidth = 5)

plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#Standard error in betas
np.std(beta_boot, axis = 0)/np.sqrt(1000)
#array([0.07311289, 0.03190114, 0.03872068])

#Standard error in probability for each year
plt.hist(np.std(prob[:,1,:], axis=1)/np.sqrt(1000),bins=50)
plt.plot(np.std(prob[:,1,:], axis=1)/np.sqrt(1000))