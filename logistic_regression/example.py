# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:58:00 2020

@author: jlamon02 (Jonathan Lamontagne) and Jared Smith
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

#Figure 3
#EDA with real axes - precip multiplied by historical avg. transformation
plt.figure()
plt.scatter(X[Y==0,1],X[Y==0,3]*150.6115385, label = 'Not Large Flood')
plt.scatter(X[Y==1,1],X[Y==1,3]*150.6115385, label = 'Large Flood')
plt.ylabel('GP/BL Nov.-Apr. Precip. (mm)')
plt.legend()
plt.xlabel('Fort Verm. DDF')

plt.figure()
plt.scatter(X[Y==0,0],X[Y==0,3]*150.6115385)
plt.scatter(X[Y==1,0],X[Y==1,3]*150.6115385)
plt.ylabel('GP/BL Nov.-Apr. Precip. (mm)')
plt.xlabel('Fort Chip. DDF')

plt.figure()
plt.scatter(X[Y==0,5],X[Y==0,3]*150.6115385)
plt.scatter(X[Y==1,5],X[Y==1,3]*150.6115385)
plt.ylabel('GP/BL Nov.-Apr. Precip. (mm)')
plt.xlabel('Freeze-up Stage (m) (Beltaos, 2014)')

plt.figure()
plt.scatter(X[Y==0,6],X[Y==0,3]*150.6115385)
plt.scatter(X[Y==1,6],X[Y==1,3]*150.6115385)
plt.ylabel('GP/BL Nov.-Apr. Precip. (mm)')
plt.xlabel('Melt Test (days)')

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

#Looking at seperation on normalized variables
plt.figure()
plt.scatter(X[Y==0,2],Y[Y==0])
plt.scatter(X[Y==1,2],Y[Y==1])
plt.xlabel('Fort Verm DDF')

plt.figure()
plt.scatter(X[Y==0,4],Y[Y==0])
plt.scatter(X[Y==1,4],Y[Y==1])
plt.xlabel('GP/BL Nov.-Apr. Precip.')

plt.figure()
plt.scatter(X[Y==0,2],X[Y==0,4])
plt.scatter(X[Y==1,2],X[Y==1,4])
plt.ylabel('GP/BL Nov.-Apr. Precip.')
plt.xlabel('Fort Verm. DDF')

plt.figure()
plt.scatter(X[Y==0,1],X[Y==0,4])
plt.scatter(X[Y==1,1],X[Y==1,4])
plt.ylabel('GP/BL Nov.-Apr. Precip.')
plt.xlabel('Fort Chip. DDF')

plt.figure()
plt.scatter(X[Y==0,6],X[Y==0,4])
plt.scatter(X[Y==1,6],X[Y==1,4])
plt.ylabel('GP/BL Nov.-Apr. Precip.')
plt.xlabel('Freeze-up Stage (Beltaos, 2014)')

plt.figure()
plt.scatter(X[Y==0,7],X[Y==0,4])
plt.scatter(X[Y==1,7],X[Y==1,4])
plt.ylabel('GP/BL Nov.-Apr. Precip.')
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
X_hold1 = np.concatenate((X[:,[0,2,4]],cross),axis=1)
res_cross = fit_logistic(X_hold1,Y)
resBase = copy.deepcopy(res_cross)
res_cross_Firth = fit_logistic(X_hold1,Y,Firth=True,resBase=resBase)
del resBase
#Note: interaction term is not significant with profile likelihood p-value = .134
#Best model is BL-GP & Fort Vermillion DDF.
#p-values for best model from profile likelihood in R:
#Int: 1.252043e-11     B1: 1.532832e-02     B2: 1.969853e-03

#Constant and Interaction Only with Firth regression
X_hold2 = np.concatenate((X[:,[0]],cross),axis=1)
resBase = copy.deepcopy(res_cross)
res_crossOnly_Firth = fit_logistic(X_hold2,Y,Firth=True,resBase=resBase)
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

#Contour plot of the best regression model over the 2 perdictors
evalptsx, evalptsy = np.meshgrid(np.arange(-3,3.2,0.2),np.arange(-3,3.2,0.2), sparse=False)
evalvals = np.exp(res_Firth.params[0] + evalptsx * res_Firth.params[1] + evalptsy * res_Firth.params[2])/(1+np.exp(res_Firth.params[0] + evalptsx * res_Firth.params[1] + evalptsy * res_Firth.params[2]))
plt.figure()
plt.contourf(evalptsx,evalptsy,evalvals, levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], cmap = 'viridis')
plt.scatter(X[Y==0,2],X[Y==0,4], color = 'black')
plt.scatter(X[Y==1,2],X[Y==1,4], color = 'orange')
plt.ylabel('GP/BL Nov.-Apr. Precip.')
plt.xlabel('Fort Verm. DDF')
plt.colorbar()

#Plot showing where 1996-97 floods are. One is low prob.
plt.figure()
plt.contourf(evalptsx,evalptsy,evalvals, levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], cmap = 'viridis')
plt.scatter(X[Y==0,2],X[Y==0,4], color = 'black')
plt.scatter(X[Y==1,2],X[Y==1,4], color = 'orange')
plt.scatter(X[[30,31],2],X[[30,31],4], color = 'green')
plt.ylabel('GP/BL Nov.-Apr. Precip.')
plt.xlabel('Fort Verm. DDF')
plt.colorbar()

#Now bootstrap#################################################################
resBase = copy.deepcopy(res_GLM)
#reload data for use in boot_master function 
# (removes the constant from X, normalization is completed in the boot_master function)
[years_boot,Y_boot,X_boot] = load_data()
[years_boot,Y_boot,X_boot] = clean_dats(years_boot,Y_boot,X_boot,column=[0,1,3,5,6],fill_years = False)
[beta_boot,bootstrap_X,bootstrap_Y] = boot_master(X_boot,Y_boot,columns=[1,3],M=1000,param=True,res=res_Firth, Firth=True, resBase=resBase)
#Alternative parametric bootstrap using MVN dist. Does not get tails of dist.
#boot_betas,bootstrap_X,bootstrap_Y=mimic_bootstrap(res_GLM,X_boot,Y_boot,M_boot=50)

#Figure 5
#pair plot of the beta empirical distribution
betas_boot_df = pd.DataFrame(data=beta_boot, columns = ['Constant', 'DDF Fort Verm.', 'GP/BL Precip.'])
sb.set_style('darkgrid')
sb.pairplot(betas_boot_df, diag_kind = 'kde', plot_kws={"s":13})

#How many points are excluded when axes are trimmed
ExcludedPtsPct = (len(np.where((beta_boot[:,0] < -15))[0]) + len(np.where((beta_boot[:,1] < -6))[0]) + len(np.where((beta_boot[:,2] > 6))[0]) 
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6))[0])
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,2] > 6))[0])
- len(np.where((beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0])
+ len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0]))/1000*100

#Save betas for plotting in another package
betas_boot_df.to_csv("betas_boot.csv", index=False)

#Standard error in betas
BetaStdErr = np.std(beta_boot, axis = 0)/np.sqrt(1000)
#array([0.07311289, 0.03190114, 0.03872068])


#Compute IJF probabilities for historical record using the bootstrapped betas
[years_All,Y_All,X_All] = load_data()
[years_All,Y_All,X_All] = clean_dats(years_All,Y_All,X_All,column=[1,3],fill_years = True)
#Normalize using only the regression years
X_All[:,1] = (X_All[:,1]-np.mean(bootstrap_X[:,1,0],0))/np.std(bootstrap_X[:,1,0],0)
X_All[:,3] = (X_All[:,3]-np.mean(bootstrap_X[:,3,0],0))/np.std(bootstrap_X[:,3,0],0)
X_All = add_constant(X_All,Y_All)
Y_HistPred = np.exp(res_Firth.params[0] + X_All[:,2] * res_Firth.params[1] + X_All[:,4] * res_Firth.params[2])/(1+np.exp(res_Firth.params[0] + X_All[:,2] * res_Firth.params[1] + X_All[:,4] * res_Firth.params[2]))
Y_HistBootPred = np.zeros([len(X_All[:,0]),len(betas_boot_df.iloc[:,0])])
for i in range(len(betas_boot_df.iloc[:,0])):
    Y_HistBootPred[:,i] = np.exp(np.array(betas_boot_df.iloc[i,0]) + X_All[:,2] * np.array(betas_boot_df.iloc[i,1]) + X_All[:,4] * np.array(betas_boot_df.iloc[i,2]))/(1+np.exp(np.array(betas_boot_df.iloc[i,0]) + X_All[:,2] * np.array(betas_boot_df.iloc[i,1]) + X_All[:,4] * np.array(betas_boot_df.iloc[i,2])))
plt.figure()
plt.plot(years_All,Y_HistPred,linewidth=2, label='Predicted - Obs. Betas', ls = '--', marker = 'D')
plt.scatter(years_All,Y_All, label = 'Floods', marker = 'o', c = 'r')
plt.ylabel('Prob. of Flood')
plt.xlabel('Year')
plt.legend(loc = 'center')

#Now GCM#######################################################################
#Load the dam filling years for plotting purposes
[years_GCM,Y_GCM,X_GCM] = load_data()
[years_GCM,Y_GCM,X_GCM] = clean_dats(years_GCM,Y_GCM,X_GCM,column=[0,1,3,5,6],fill_years = True)

Temp_GCM,Precip_GCM,Years_GCM = load_GCMS(X_GCM[:,1],X_GCM[:,3])
prob,flood,cum_flood,waits = simulate_GCM_futures(Y_GCM,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=1000,N_prob=1000)
#For use with MVN dist. parametric bootstrap
#prob,flood,cum_flood,waits = simulate_GCM_futures(Y_GCM,bootstrap_X,bootstrap_Y,boot_betas,Temp_GCM,Precip_GCM,M_boot=5000,N_prob=1000)

#Standard error in probability for each year
plt.hist(np.std(prob[:,1,:], axis=1)/np.sqrt(1000),bins=50)
#plt.plot(np.std(prob[:,1,:], axis=1)/np.sqrt(1000))

#Run analysis for the GCM temperature and precip over the historical record.
#Using Nprob = 1 because only historical info is needed
Temp_GCM_NoHistSplice,Precip_GCM_NoHistSplice,Years_GCM_NoHistSplice = load_GCMS(X_GCM[:,1],X_GCM[:,3],histSplice=False)
prob_nhs,flood_nhs,cum_flood_nhs,waits_nhs = simulate_GCM_futures(Y_GCM,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM_NoHistSplice,Precip_GCM_NoHistSplice,M_boot=1000,N_prob=1)

#Also run without splicing in historical floods (use probability of flood to generate distribution of floods in historical years)
prob_nhsf,flood_nhsf,cum_flood_nhsf,waits_nhsf = simulate_GCM_futures(Y_GCM,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=1000,N_prob=1000, histSplice=False)

#Table 5
#Wait time percentiles from data and from the probability of flood over historical record
wait1962obs = np.percentile(waits[0,:,:,:], q=[2.5,25,50,75,97.5])
wait1962pred = np.percentile(waits_nhsf[0,:,:,:], q=[2.5,25,50,75,97.5])
wait1990obs = np.percentile(waits[28,:,:,:], q=[2.5,25,50,75,97.5])
wait1990pred = np.percentile(waits_nhsf[28,:,:,:], q=[2.5,25,50,75,97.5])
wait2010obs = np.percentile(waits[48,:,:,:], q=[2.5,25,50,75,97.5])
wait2010pred = np.percentile(waits_nhsf[48,:,:,:], q=[2.5,25,50,75,97.5])
del prob_nhsf, flood_nhsf, cum_flood_nhsf, waits_nhsf

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
del scenario, plt_perc, plt_perc2, percentiles, i
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
del scenario, plt_perc, plt_perc2, percentiles, i
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
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER Revision = Moving Average in Real Space with a plot in log scale, backwards looking, correct CI on 20-yr average
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window)
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2
# PAPER Revision = Moving Average in Real Space with a plot in log scale, centered, correct CI on 20-yr average
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,plt_perc2,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window,years=np.array(range(1961,2100)[int(window/2):int(len(range(1961,2100))-window/2)]), CIind=1, xlim=[1960,2100])
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2
#Figure 8
# PAPER Revision = Moving Average in Real Space with a plot in log scale, centered, return period second y axis, correct CI on 20-yr average
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,1/plt_perc2,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window,years=np.array(range(1961,2100)[int(window/2):int(len(range(1961,2100))-window/2)]), CIind=1, xlim=[1960,2100], ylim2=[1/0.0001, 1], twoYax=True,ylabel2='IJF Return Period')
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2
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
    percentile_fill_plot_single(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green', xlim=[1980,2100])
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue', xlim=[1980,2100])
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER Revision = Moving Average in Real Space with a plot in log scale, backward looking, correct CI on 20-yr average
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_single(plt_perc,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green', years=np.array(range(1961,2100)[window:int(len(range(1961,2100)))]), xlim=[1980,2100])
    percentile_fill_plot_single(plt_perc2,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue', years=np.array(range(1961,2100)[window:int(len(range(1961,2100)))]), xlim=[1980,2100])
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2
#Figure 9
# PAPER Revision = Moving Average in Real Space with a plot in log scale, centered, correct CI on 20-yr average
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_single(plt_perc,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green', years=np.array(range(1961,2100)[int(window/2):int(len(range(1961,2100))-window/2)]), xlim=[1960,2100])
    percentile_fill_plot_single(plt_perc2,title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue', years=np.array(range(1961,2100)[int(window/2):int(len(range(1961,2100))-window/2)]), xlim=[1960,2100])
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

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
del scenario, plt_perc, plt_perc2, percentiles, i
# PRESENTATION = Moving Average in Real Space with a plot in real space
window=10
percentiles = [2.5,25,50,75,97.5]
#Probability
plt_perc = np.percentile(Y_HistBootPred,percentiles,axis=1)
for i in range(5):
    plt_perc[i,0:(np.shape(years_All)[0]-(window-1))]=moving_average(plt_perc[i,:],window)    
percentile_fill_plot_single(plt_perc[:,31:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1949,end=2021,xlim=[1950,2020],Ypobs = Y_HistPred[31:103])
del plt_perc, percentiles, i
#With observed data
percentiles = [2.5,25,50,75,97.5]
#Probability
plt_perc = np.percentile(Y_HistBootPred,percentiles,axis=1)
for i in range(5):
    plt_perc[i,0:(np.shape(years_All)[0]-(window-1))]=moving_average(plt_perc[i,:],window)    
percentile_fill_plot_single(plt_perc[:,31:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1949,end=2021,xlim=[1950,2020],Yobs=Y_All[31:103], Ypobs = Y_HistPred[31:103])
del plt_perc, percentiles, i
# PRESENTATION = Moving Average in Real Space with a plot in real space Correct CIs for 20 year moving average
#Y moving average
YMA = np.zeros([np.shape(years_All)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
for i in range(len(betas_boot_df.iloc[:,0])):
    YMA[:,i] = moving_average(Y_HistBootPred[:,i],window)
percentiles = [2.5,25,50,75,97.5]
#Probability
plt_perc = np.percentile(YMA,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,35:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1953,end=2021,xlim=[1950,2020],Ypobs = Y_HistPred[35:103])
#With observed data - backward looking
percentile_fill_plot_single(plt_perc[:,35:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1953,end=2021,xlim=[1950,2020],Yobs=Y_All[35:103], Ypobs = Y_HistPred[35:103])
#Figure 6
#PAPER - Revision: With observed data - centered
percentile_fill_plot_single(plt_perc[:,40:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1953,end=2016,xlim=[1950,2020],Yobs=Y_All[40:103], Ypobs = Y_HistPred[40:103])
##All data to 1915
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1915,end=2021,xlim=[1910,2020],Ypobs = Y_HistPred,years=years_All[(window-1):(np.shape(years_All)[0])])
#With observed data: large floods only
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1915,end=2021,xlim=[1910,2020],Yobs=Y_All, Ypobs = Y_HistPred,years=years_All[(window-1):(np.shape(years_All)[0])])
#With observed data: large + moderate floods
Y_AllLM = np.loadtxt('cleaned_dataLMS.csv',delimiter=',',skiprows=1,usecols=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1915,end=2021,xlim=[1910,2020],Yobs=Y_AllLM, Ypobs = Y_HistPred,years=years_All[(window-1):(np.shape(years_All)[0])])
#With observed data: large + moderate + small floods
Y_AllLMS = np.loadtxt('cleaned_dataLMS.csv',delimiter=',',skiprows=1,usecols=2)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_All)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='Bootstrapped 50% and 95% CIs',window=window,start=1915,end=2021,xlim=[1910,2020],Yobs=Y_AllLMS, Ypobs = Y_HistPred,years=years_All[(window-1):(np.shape(years_All)[0])])
del plt_perc, percentiles, i

#Figure S.2
#PAPER Revision
window=1
#Y moving average
YMA = np.zeros([np.shape(years_All)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
for i in range(len(betas_boot_df.iloc[:,0])):
    YMA[:,i] = moving_average(Y_HistBootPred[:,i],window)
percentiles = [2.5,25,50,75,97.5]
#Probability
plt_perc = np.percentile(YMA,percentiles,axis=1)
#With observed data - centered
percentile_fill_plot_single(plt_perc[:,44:(np.shape(years_All)[0]-(window-1))],title='IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[-0.05,1.05],Names='Bootstrapped 50% and 95% CIs',window=window,start=1962,end=2021,xlim=[1960,2022],Yobs=Y_All[44:103], Ypobs = Y_HistPred[44:103])

window=20
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
del scenario, plt_perc, plt_perc2, percentiles, i

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
del scenario, plt_perc, plt_perc2, percentiles, i
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
del scenario, plt_perc, plt_perc2, percentiles, i
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
del scenario, plt_perc, plt_perc2, percentiles, i

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
del scenario, plt_perc, plt_perc2, percentiles, i

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
del scenario, plt_perc, plt_perc2, percentiles, i
#Graphical Abstract
#PAPER Revision = Return period plot for 50% CI using moving average in real space, centered, correct CIs on 20-yr average
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    probMA = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    probMA2 = np.zeros([np.shape(Temp_GCM)[0]-(window-1),len(betas_boot_df.iloc[:,0])])
    for i in range(len(betas_boot_df.iloc[:,0])):
        probMA[:,i] = moving_average(prob[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(1/plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],1/plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Return Period',scale='log',ylim=[1,10000],Names=RCPs,window=window, years=np.array(range(1961,2100)[int(window/2):int(len(range(1961,2100))-window/2)]), CIind=1, xlim=[1960,2100])
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

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
del scenario, plt_perc, plt_perc2, percentiles, i
#PRESENTATION = Return period plot for 50% CI using moving average in real space, real axis, 2050
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
    percentile_fill_plot_double(1/plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],1/plt_perc2[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Return Period',scale='linear',ylim=[0,100],Names=RCPs,window=window,xlim=[1980,2050],CIind=1)
del scenario, plt_perc, plt_perc2, percentiles, i

#All GCMs on one plot using moving average of the median
plt_perc = np.zeros([np.shape(Temp_GCM)[1],np.shape(Temp_GCM)[0]])
for scenario in range(np.shape(Temp_GCM)[1]):
    
    #Probability
    plt_perc[scenario,:] = np.percentile(prob[:,scenario,:],50,axis=1)

    plt_perc[scenario,0:(np.shape(Temp_GCM)[0]-(window-1))]=moving_average(plt_perc[scenario,:],window)
    #for j in range((np.shape(Temp_GCM)[0]-(window-1)),np.shape(Temp_GCM)[0]):
    #    plt_perc[scenario,j] = np.mean(plt_perc[scenario,j:np.shape(Temp_GCM)[0]])
percentile_plot_single(plt_perc[:,0:(np.shape(Temp_GCM)[0]-(window-1))],title='Ice Jam Flood Projection',ylabel='IJF Probability',scale='log',ylim=[0.00001,0.5],split='gcm',Names=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR'],window=window)
del scenario, plt_perc

#Plot GCM Data
#Temp RCPs
plt.figure()
for i in range(6):
    plt.plot(range(1962,2091),moving_average(Temp_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1962,2091),moving_average(Temp_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Fort Verm DDF')
del i

#Prec RCPs
plt.figure()
for i in range(6):
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,i],10),'b',linewidth=3,label=RCPs[0])
    plt.plot(range(1962,2091),moving_average(Precip_GCM[:,6+i],10),'r',linewidth=3,label=RCPs[1])
plt.legend(RCPs)
plt.xlabel('Year')
plt.ylabel('Beaverlodge Precip')
del i

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
del i

#Plot of the historical probabilities for each model
#RCP85
for i in range(6):
    plt.figure()
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],95,axis=1),linewidth=2, label='95%', c = 'r', ls = '--', marker = 'D')
    plt.plot(Years_GCM[0:59],np.average(prob_nhs[0:59,i,:],axis=1),linewidth=2, label='Mean', marker = 'o')
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],5,axis=1),linewidth=2, label='5%', c = 'g', ls = '--', marker = 'D')
    plt.scatter(Years_GCM[0:59],np.concatenate((Y_GCM[0:6],[0,0,0,0],Y_GCM[6:55])), label = 'Floods', marker = 'o', c = 'c')
    plt.ylabel('Prob. of Flood')
    plt.xlabel('Year')
    plt.legend(loc = 'center')
    plt.title(GCMs[i] + ' - RCP85')
    plt.savefig(GCMs[i] + '_RCP85.png')
del i
#RCP45  
for i in range(6,12):
    plt.figure()
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],95,axis=1),linewidth=2, label='95%', c = 'r', ls = '--', marker = 'D')
    plt.plot(Years_GCM[0:59],np.average(prob_nhs[0:59,i,:],axis=1),linewidth=2, label='Mean', marker = 'o')
    plt.plot(Years_GCM[0:59],np.percentile(prob_nhs[0:59,i,:],5,axis=1),linewidth=2, label='5%', c = 'g', ls = '--', marker = 'D')
    plt.scatter(Years_GCM[0:59],np.concatenate((Y_GCM[0:6],[0,0,0,0],Y_GCM[6:55])), label = 'Floods', marker = 'o', c = 'c')
    plt.ylabel('Prob. of Flood')
    plt.xlabel('Year')
    plt.legend(loc = 'center')
    plt.title(GCMs[i-6] + ' - RCP45')
    plt.savefig(GCMs[i-6] + '_RCP45.png')
del i

#Survival Analysis#############################################################
median = survival(waits)
AvgMedWait = np.zeros([np.shape(Temp_GCM)[0]-10+1,np.shape(Temp_GCM)[1]])
for i in range(np.shape(Temp_GCM)[1]):
    AvgMedWait[:,i] = moving_average(median[:,i],n=10)
del i

#Table 5
GCM2030Waits = median[68,:]
GCM2050Waits = median[88,:]

plt.figure()
for i in range(6):
    plt.plot(Years_GCM[38:89],median[38:89,i],'g',linewidth = 5)
    plt.plot(Years_GCM[38:89],median[38:89,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#to 2070 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(Years_GCM[38:109],median[38:109,i],'g',linewidth = 5)
    plt.plot(Years_GCM[38:109],median[38:109,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#to 2100 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(Years_GCM,median[:,i],'g',linewidth = 5)
    plt.plot(Years_GCM,median[:,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)


#Cumulative floods
plt.figure()
plt.plot(Years_GCM[:],np.mean(cum_flood[:,0,:,:],axis=1),'g',linewidth = 0.2)
plt.xlabel('Year')
plt.ylabel('Cumulative Floods')
plt.title(GCMs[0])
plt.legend([RCPs[0]])
plt.ylim=[0,40]

plt.figure()
plt.plot(Years_GCM[:],np.mean(cum_flood[:,6,:,:],axis=1),'b',linewidth = 0.2)
plt.xlabel('Year')
plt.ylabel('Cumulative Floods')
plt.title(GCMs[0])
plt.legend([RCPs[1]])
plt.ylim=[0,40]

#### Regression Models with Peace Point and Peace River for SI
# Load data
[years_PR,Y_PR,X_PR] = load_data(PPPR=True)
[years_PR,Y_PR,X_PR] = clean_dats(years_PR,Y_PR,X_PR,column=[0,1,3,5,6,9,10],fill_years = False)

[years_PP,Y_PP,X_PP] = load_data(PPPR=True)
[years_PP,Y_PP,X_PP] = clean_dats(years_PP,Y_PP,X_PP,column=[0,1,3,5,6,11,12],fill_years = False)

X_PR = normalize(X_PR)
X_PP = normalize(X_PP)

X_PR = add_constant(X_PR,Y_PR)
X_PP = add_constant(X_PP,Y_PP)

# Iterative model building
#  All two parameter models (constant + variable)
#   Firth two parameter models
[betasf_PR, pvaluesf_PR,aicf_PR,aiccf_PR,bicf_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0],Firth=True)
[betasf_PP, pvaluesf_PP,aicf_PP,aiccf_PP,bicf_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0],Firth=True)
# Three parameter models
#  Firth three parameter models
[betas3f_PR, pvalues3f_PR,aic3f_PR,aicc3f_PR,bic3f_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,4],Firth=True)
[betas3f_PP, pvalues3f_PP,aic3f_PP,aicc3f_PP,bic3f_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,4],Firth=True)
#  Best Freeze-up model with three parameters
[betas3frf_PR, pvalues3frf_PR,aic3frf_PR,aicc3frf_PR,bic3frf_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,6],Firth=True)
[betas3frf_PP, pvalues3frf_PP,aic3frf_PP,aicc3frf_PP,bic3frf_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,6],Firth=True)
# Four parameter models with Firth regression
[betas4f_PR, pvalues4f_PR,aic4f_PR,aicc4f_PR,bic4f_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,2,4],Firth=True)
[betas4f_PP, pvalues4f_PP,aic4f_PP,aicc4f_PP,bic4f_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,2,4],Firth=True)
#  Best with Freezeup
[betas4frf_PR, pvalues4frf_PR,aic4frf_PR,aicc4frf_PR,bic4frf_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,4,6],Firth=True)
[betas4frf_PP, pvalues4frf_PP,aic4frf_PP,aicc4frf_PP,bic4frf_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,4,6],Firth=True)
# Five parameter models - Same models selected for both freeze-up and regular at this stage.
[betas5f_PR, pvalues5f_PR,aic5f_PR,aicc5f_PR,bic5f_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,2,4,10],Firth=True)
[betas5f_PP, pvalues5f_PP,aic5f_PP,aicc5f_PP,bic5f_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,2,4,13],Firth=True)
#  Best with Freezeup
[betas5frf_PR, pvalues5frf_PR,aic5frf_PR,aicc5frf_PR,bic5frf_PR]=iterate_logistic(X_PR,Y_PR, fixed_columns = [0,2,4,6],Firth=True)
[betas5frf_PP, pvalues5frf_PP,aic5frf_PP,aicc5frf_PP,bic5frf_PP]=iterate_logistic(X_PP,Y_PP, fixed_columns = [0,2,4,6],Firth=True)