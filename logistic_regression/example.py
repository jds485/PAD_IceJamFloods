# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:58:00 2020

@author: jlamon02
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
[years,Y,X] = load_data()
[years,Y,X] = clean_dats(years,Y,X,column=[0,1,3,5,6])

X = normalize(X)

X = add_constant(X,Y)

#Test constant model with GLM vs. logistic regression
res_GLM = fit_logistic(X[:,[0,1,4]],Y)

#Test Firth regression method
import statsmodels.api as sm
res_Firth = fit_logistic(X[:,[0,1,4]],Y,Firth=True)

logit=sm.Logit(Y,X[:,[0,1,4]])
res_SM = logit.fit()

#Jared: Betas and aic seem the same.  P-values are the same out a ways.  bic are different.
# I belived the SM bic.  Everything else looks consistent between the two.
#Looking at seperation
plt.figure()
plt.scatter(X[Y==0,1],Y[Y==0])
plt.scatter(X[Y==1,1],Y[Y==1])
plt.xlabel('Fort Chip DDF')

plt.figure()
plt.scatter(X[Y==0,4],Y[Y==0])
plt.scatter(X[Y==1,4],Y[Y==1])
plt.xlabel('BL Precip')

plt.figure()
plt.scatter(X[Y==0,4],X[Y==0,1])
plt.scatter(X[Y==1,4],X[Y==1,1])
plt.xlabel('BL Precip')
plt.ylabel('Fort Chip DDF')

# Example of iterative model building
[betas, pvalues,aic,bic]=iterate_logistic(X,Y, fixed_columns = [0])
# Three parameter models
[betas3, pvalues3,aic3,bic3]=iterate_logistic(X,Y, fixed_columns = [0,4])
# Cross parameter models
cross=np.reshape(X[:,1]*X[:,4],(-1,1))
X_hold = np.concatenate((X[:,[0,1,4]],cross),axis=1)
res_cross = fit_logistic(X_hold,Y)
#Best model is BL-GP & Fort Vermillion DDF.

#Now bootstrap
#[beta_boot,bootstrap_X,bootstrap_Y] = boot_master(X,Y,columns=[0,1,4],M=5000,block_length = 5)

