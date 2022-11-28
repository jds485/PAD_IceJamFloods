# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:06:32 2022

@author: js4yd
"""

#Compare the Bayesian beta distribution to the logistic parametric bootstrap
import os
os.chdir('../../../logistic_regression')

#Copy from Lamontagne et al
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
evalvals = (np.exp(res_Firth.params[0] + evalptsx * res_Firth.params[1] + 
                   evalptsy * res_Firth.params[2])/
    (1+np.exp(res_Firth.params[0] + evalptsx * res_Firth.params[1] + 
              evalptsy * res_Firth.params[2])))
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

#Load beta_boot with Bayesian estimated betas
beta_boot_Bayes = np.loadtxt('../bayesian_regression/DREAMzs_L15_VermPrecip_1962-2020/BayesBetas_hold.csv',delimiter=',',skiprows=1)
beta_boot_BayesLg = np.loadtxt('../bayesian_regression/DREAMzs_L15_VermPrecip_1962-2020/BayesBetasLg_hold.csv',delimiter=',',skiprows=1)

#pair plot of the beta empirical distribution
betas_boot_df = pd.DataFrame(data=beta_boot, columns = ['Constant', 'DDF Fort Verm.', 'GP/BL Precip.'])
betas_boot_Bayes_df = pd.DataFrame(data=beta_boot_Bayes, columns = ['Constant', 'DDF Fort Verm.', 'GP/BL Precip.'])
betas_boot_BayesLg_df = pd.DataFrame(data=beta_boot_BayesLg, columns = ['Constant', 'DDF Fort Verm.', 'GP/BL Precip.'])
sb.set_style('darkgrid')
sb.pairplot(betas_boot_df, diag_kind = 'kde', plot_kws={"s":13})
sb.pairplot(betas_boot_Bayes_df, diag_kind = 'kde', plot_kws={"s":13})
sb.pairplot(betas_boot_BayesLg_df, diag_kind = 'kde', plot_kws={"s":13})

#Percentiles - extremes more similar than IQR
np.percentile(beta_boot, [2.5,25,75,97.5],axis=0)
np.percentile(beta_boot_Bayes, [2.5,25,75,97.5],axis=0)
np.percentile(beta_boot_BayesLg, [2.5,25,75,97.5],axis=0)

#How many points are excluded when axes are trimmed
ExcludedPtsPct = (len(np.where((beta_boot[:,0] < -15))[0]) + len(np.where((beta_boot[:,1] < -6))[0]) + len(np.where((beta_boot[:,2] > 6))[0]) 
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6))[0])
- len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,2] > 6))[0])
- len(np.where((beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0])
+ len(np.where((beta_boot[:,0] < -15) & (beta_boot[:,1] < -6) & (beta_boot[:,2] > 6))[0]))/1000*100

#Compute IJF probabilities for historical record using the bootstrapped betas
[years_All,Y_All,X_All] = load_data()
[years_All,Y_All,X_All] = clean_dats(years_All,Y_All,X_All,column=[1,3],fill_years = True)
#Normalize using only the regression years
X_All[:,1] = (X_All[:,1]-np.mean(bootstrap_X[:,1,0],0))/np.std(bootstrap_X[:,1,0],0)
X_All[:,3] = (X_All[:,3]-np.mean(bootstrap_X[:,3,0],0))/np.std(bootstrap_X[:,3,0],0)
X_All = add_constant(X_All,Y_All)
Y_HistPred = (np.exp(res_Firth.params[0] + 
                     X_All[:,2] * res_Firth.params[1] + 
                     X_All[:,4] * res_Firth.params[2])/
    (1+np.exp(res_Firth.params[0] + 
              X_All[:,2] * res_Firth.params[1] + 
              X_All[:,4] * res_Firth.params[2])))
Y_HistBootPred = np.zeros([len(X_All[:,0]),len(betas_boot_Bayes_df.iloc[:,0])])
for i in range(len(betas_boot_Bayes_df.iloc[:,0])):
    Y_HistBootPred[:,i] = (np.exp(np.array(betas_boot_Bayes_df.iloc[i,0]) + 
                  X_All[:,2] * np.array(betas_boot_Bayes_df.iloc[i,1]) + 
                  X_All[:,4] * np.array(betas_boot_Bayes_df.iloc[i,2]))/
        (1+np.exp(np.array(betas_boot_Bayes_df.iloc[i,0]) + 
                  X_All[:,2] * np.array(betas_boot_Bayes_df.iloc[i,1]) + 
                  X_All[:,4] * np.array(betas_boot_Bayes_df.iloc[i,2]))))
plt.figure()
plt.plot(years_All,Y_HistPred,linewidth=2, label='Predicted - Obs. Betas', ls = '--', marker = 'D')
plt.scatter(years_All,Y_All, label = 'Floods', marker = 'o', c = 'r')
plt.ylabel('Prob. of Flood')
plt.xlabel('Year')
plt.legend(loc = 'center')