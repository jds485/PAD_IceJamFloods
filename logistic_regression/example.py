# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:58:00 2020

@author: jlamon02
"""

from utils import *
from utils_figures import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import seaborn as sb
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
# set seed
random.seed(123020)
np.random.seed(123020)

# Load data
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = False)

#Evaluate square root of temperature as predictor with GP/BL precip
X_sqrt = np.concatenate((X[:,[0,1,3]],np.sqrt(-X[:,[0,1]])),axis=1)

X = normalize(X)
X_sqrt = normalize(X_sqrt)

X = add_constant(X,Y)
X_sqrt = add_constant(X_sqrt,Y)

#Test constant model with GLM vs. logistic regression
res_GLM = fit_logistic(X[:,[0,2,4]],Y)

resBase = copy.deepcopy(res_GLM)
#Firth regression method
res_Firth = fit_logistic(X[:,[0,2,4]],Y,Firth=True, resBase = resBase)
del resBase
#Compare with the statsmodel logit function
#Betas and aic seem the same.  P-values are the same out a ways.  bic are different.
# SM bic is correct upon investigation. Returning that BIC in functions.
logit=sm.Logit(Y,X[:,[0,2,4]])
res_SM = logit.fit()

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
plt.scatter(X[Y==0,6],X[Y==0,4])
plt.scatter(X[Y==1,6],X[Y==1,4])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Freeze-up Stage (Beltaos, 2014)')

plt.figure()
plt.scatter(X[Y==0,7],X[Y==0,4])
plt.scatter(X[Y==1,7],X[Y==1,4])
plt.ylabel('BL/GP Nov.-Apr. Precip.')
plt.xlabel('Melt Test')

# Iterative model building
#  All two parameter models (constant + variable)
[betas, pvalues,aic,aicc,bic]=iterate_logistic(X,Y, fixed_columns = [0])
#   Firth two parameter models
[betasf, pvaluesf,aicf,aiccf,bicf]=iterate_logistic(X,Y, fixed_columns = [0],Firth=True)
# Three parameter models
[betas3, pvalues3,aic3,aicc3,bic3]=iterate_logistic(X,Y, fixed_columns = [0,4])
#  Firth three parameter models
[betas3f, pvalues3f,aic3f,aicc3f,bic3f]=iterate_logistic(X,Y, fixed_columns = [0,4],Firth=True)
#  Best Freeze-up model with three parameters
[betas3frf, pvalues3frf,aic3frf,aicc3frf,bic3frf]=iterate_logistic(X,Y, fixed_columns = [0,6],Firth=True)
# Cross parameter models for X2 and X4 (best model without freezeup)
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

#Constant and Interaction Only with Firth regression
X_hold = np.concatenate((X[:,[0]],cross),axis=1)
resBase = copy.deepcopy(res_cross)
res_crossOnly_Firth = fit_logistic(X_hold,Y,Firth=True,resBase=resBase)
del resBase

#Constant and Principal Components (X2 and X4 only) with Firth regression
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X[:,[2,4]])
principalComponents = np.concatenate((X[:,[0]],principalComponents),axis=1)
resBase = copy.deepcopy(res_GLM)
#  Only 1st PC
res_PC1_Firth = fit_logistic(principalComponents[:,[0,1]],Y,Firth=True,resBase=resBase)
del resBase
resBase = copy.deepcopy(res_GLM)
#  Both PC
res_PC_Firth = fit_logistic(principalComponents,Y,Firth=True,resBase=resBase)
del resBase

#Plot of the PC model and the preferred 2 parameter model
plt.figure()
plt.plot(years,res_PC1_Firth.predict,linewidth=2, label='PC1', ls = '--', marker = 'D')
plt.plot(years,res_Firth.predict,linewidth=2, label='GP + FV', ls = '--', marker = '^')
plt.scatter(years,Y, label = 'Floods', marker = 'o', c = 'r')
plt.ylabel('Prob. of Flood')
plt.xlabel('Year')
plt.legend(loc = 'center')

# Four parameter models with Firth regression
[betas4f, pvalues4f,aic4f,aicc4f,bic4f]=iterate_logistic(X,Y, fixed_columns = [0,2,4],Firth=True)
#  Best with Freezeup
[betas4frf, pvalues4frf,aic4frf,aicc4frf,bic4frf]=iterate_logistic(X,Y, fixed_columns = [0,4,6],Firth=True)
# Five parameter models
[betas5f, pvalues5f,aic5f,aicc5f,bic5f]=iterate_logistic(X,Y, fixed_columns = [0,2,4,8],Firth=True)
#  Best with Freezeup
[betas5frf, pvalues5frf,aic5frf,aicc5frf,bic5frf]=iterate_logistic(X,Y, fixed_columns = [0,2,4,6],Firth=True)

#Evaluate square root of temperature models for Fort Chip. and Fort Verm.
resBase = copy.deepcopy(res_GLM)
res_sqrtC_Firth = fit_logistic(X_sqrt[:,[0,4]],Y,Firth=True,resBase=resBase)
res_sqrtV_Firth = fit_logistic(X_sqrt[:,[0,5]],Y,Firth=True,resBase=resBase)
[betas3f_sqrt, pvalues3f_sqrt,aic3f_sqrt,aicc3f_sqrt,bic3f_sqrt]=iterate_logistic(X_sqrt,Y, fixed_columns = [0,3],Firth=True)
del resBase


#Now bootstrap#################################################################
resBase = copy.deepcopy(res_GLM)
#reload data for use in boot_master function (removes the constant from X)
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = False)
[beta_boot,bootstrap_X,bootstrap_Y] = boot_master(X,Y,columns=[1,3],M=1000,param=True,res=res_Firth, Firth=True, resBase=resBase)
#Alternative parametric bootstrap using MVN dist. Does not get tails of dist.
#boot_betas,bootstrap_X,bootstrap_Y=mimic_bootstrap(res_GLM,X,Y,M_boot=50)

#pair plot of the beta empirical distribution
betas_boot_df = pd.DataFrame(data=beta_boot, columns = ['Constant', 'DDF Fort Verm.', 'BL/GP Precip.'])
sb.set_style('darkgrid')
sb.pairplot(betas_boot_df, diag_kind = 'kde', plot_kws={"s":13})

#How many points are excluded when axes are trimmed
ExcludedPtsPct = (len(np.where((beta_boot[:,0] < -15))[0]) + len(np.where((beta_boot[:,1] < -6))[0]) + len(np.where((beta_boot[:,2] > 6))[0]) 
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6))[0])
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,2] > 6))[0])
- len(np.where((beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0])
+ len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0]))/1000*100

#Now GCM#######################################################################
#Load the dam filling years for plotting purposes
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6],fill_years = True)

Temp_GCM,Precip_GCM,Years_GCM = load_GCMS(X[:,1],X[:,3])
prob,flood,cum_flood,waits = simulate_GCM_futures(Y,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=1000,N_prob=1000)
#For use with MVN dist. parametric bootstrap
#prob,flood,cum_flood,waits = simulate_GCM_futures(Y,bootstrap_X,bootstrap_Y,boot_betas,Temp_GCM,Precip_GCM,M_boot=5000,N_prob=1000)

#Standard error in betas
BetaStdErr = np.std(beta_boot, axis = 0)/np.sqrt(1000)
#array([0.07311289, 0.03190114, 0.03872068])

#Standard error in probability for each year
plt.hist(np.std(prob[:,1,:], axis=1)/np.sqrt(1000),bins=50)
plt.plot(np.std(prob[:,1,:], axis=1)/np.sqrt(1000))

#Run analysis for the GCM temperature and precip over the historical record.
#Using Nprob = 1 because only historical info is needed
Temp_GCM_NoHistSplice,Precip_GCM_NoHistSplice,Years_GCM_NoHistSplice = load_GCMS(X[:,1],X[:,3],histSplice=False)
prob_nhs,flood_nhs,cum_flood_nhs,waits_nhs = simulate_GCM_futures(Y,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM_NoHistSplice,Precip_GCM_NoHistSplice,M_boot=1000,N_prob=1)

#Now plots#####################################################################
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
window = 20
#Make 50% confidence corridor plots (IQR) with window-year smoothing
# Moving Average in Real Space
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
# Moving Median in Real Space
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
# PAPER = Moving Average in Real Space with a plot in log scale
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
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window)
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_single(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green')
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue')
# Moving Average in Real Space with a plot in log scale with 95%CI
for scenario in range(6):
    percentiles = [10,2.5,50,97.5,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs,window=window)

# Moving Median in Real Space with a plot in log scale
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

# Moving Average in log space
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
#  With power of 10 axes
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
    percentile_fill_plot_double(10**(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))]),10**(plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))]),title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000000001,1],Names=RCPs,window=window)
#  With power of 10 axes and 95% CI
for scenario in range(6):
    percentiles = [2.5,2.5,50,97.5,97.5]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(np.log10(plt_perc[i,:]),window)
        plt_perc2[i,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(np.log10(plt_perc2[i,:]),window)
        #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
        #    plt_perc[i,j] = np.mean(plt_perc[i,j:np.shape(Temp_GCM)[0]])
        #    plt_perc2[i,j] = np.mean(plt_perc2[i,j:np.shape(Temp_GCM)[0]])
    percentile_fill_plot_double(10**(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))]),10**(plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))]),title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000000001,1],Names=RCPs,window=window)

# Moving Median in log space
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

#PAPER = Return period plot for 50% CI using moving average in real space
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
# Return period plot for 50% CI using moving median in real space
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
    percentile_fill_plot_double(1/plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],1/plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Return Period',scale='log',ylim=[1,100000],Names=RCPs,window=window)

#All GCMs on one plot using moving average of the median
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

#Plot of the historical probabilities for each model
#RCP85
for i in range(6):
    plt.figure()
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],95,axis=1),linewidth=2, label='95%', c = 'r', ls = '--', marker = 'D')
    plt.plot(Years_GCM[0:59],np.average(prob_nhs[0:59,i,:],axis=1),linewidth=2, label='Mean', marker = 'o')
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],5,axis=1),linewidth=2, label='5%', c = 'g', ls = '--', marker = 'D')
    plt.scatter(Years_GCM[0:59],np.concatenate((Y[0:6],[0,0,0,0],Y[6:55])), label = 'Floods', marker = 'o', c = 'c')
    plt.ylabel('Prob. of Flood')
    plt.xlabel('Year')
    plt.legend(loc = 'center')
    plt.title(GCMs[i] + ' - RCP85')
    plt.savefig(GCMs[i] + '_RCP85.png')
#RCP45  
for i in range(6,12):
    plt.figure()
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],95,axis=1),linewidth=2, label='95%', c = 'r', ls = '--', marker = 'D')
    plt.plot(Years_GCM[0:59],np.average(prob_nhs[0:59,i,:],axis=1),linewidth=2, label='Mean', marker = 'o')
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],5,axis=1),linewidth=2, label='5%', c = 'g', ls = '--', marker = 'D')
    plt.scatter(Years_GCM[0:59],np.concatenate((Y[0:6],[0,0,0,0],Y[6:55])), label = 'Floods', marker = 'o', c = 'c')
    plt.ylabel('Prob. of Flood')
    plt.xlabel('Year')
    plt.legend(loc = 'center')
    plt.title(GCMs[i-6] + ' - RCP45')
    plt.savefig(GCMs[i-6] + '_RCP45.png')

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

#to 2070 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(Years_GCM[38:109],median[38:109,i],'g',linewidth = 5)
    plt.plot(Years_GCM[38:109],median[38:109,i+6],'b',linewidth = 5)

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