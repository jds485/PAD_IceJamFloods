# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:32:51 2021

@author: js4yd
"""

from utils import *
from utils_figures import *
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
# set seed
random.seed(51321)
np.random.seed(51321)

#Load data from R
#Prob Z=1|X of flood
p_L15sm = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/p.csv',delimiter=',',skiprows=1)
#prob Y=1 | X of flood
q_L15sm = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/q.csv',delimiter=',',skiprows=1)
#observed data matrix 1915-1962 generated from q
Y_L15sm = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/y.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Yrep_L15sm = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/yrep.csv',delimiter=',',skiprows=1)
years_L15sm = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/years.csv',delimiter=',',skiprows=1)

#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
qMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_L15sm[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_L15sm[:,i] = moving_average(Yrep_L15sm[i,:],window)
    pMA_L15sm[:,i] = moving_average(p_L15sm[i,:],window)
    qMA_L15sm[:,i] = moving_average(q_L15sm[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])

plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])

plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])

plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))


#p and Yrep
plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)

#Y and Yrep
plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)

#p and q
plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title='10-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)


#GCM Plots
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
window=20

#Load GCM probabilities and combine into an array
p1 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_Had85.csv',delimiter=',',skiprows=1)
p2 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_Acc85.csv',delimiter=',',skiprows=1)
p3 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_Can85.csv',delimiter=',',skiprows=1)
p4 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_CCS85.csv',delimiter=',',skiprows=1)
p5 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_CNR85.csv',delimiter=',',skiprows=1)
p6 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_MPI85.csv',delimiter=',',skiprows=1)
p7 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_Had45.csv',delimiter=',',skiprows=1)
p8 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_ACC45.csv',delimiter=',',skiprows=1)
p9 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_Can45.csv',delimiter=',',skiprows=1)
p10 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_CCS45.csv',delimiter=',',skiprows=1)
p11 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_CNR45.csv',delimiter=',',skiprows=1)
p12 = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/GCMp_MPI45.csv',delimiter=',',skiprows=1)
prob = np.stack([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], axis=2)
del p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12

#Re-arrange order of array index to match function
prob = np.moveaxis(prob, [2,1], [1,2])

GCMyears = np.loadtxt('../bayesian_regression/DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s/year62t2099.csv',delimiter=',',skiprows=1)

sb.set_style('darkgrid')

# PAPER = Moving Average in Real Space with a plot in log scale
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window,years=GCMyears[(window-1):(np.shape(GCMyears)[0])], xlim=[1980,2100])
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green',years=GCMyears[(window-1):(np.shape(GCMyears)[0])])
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue',years=GCMyears[(window-1):(np.shape(GCMyears)[0])])
del scenario, plt_perc, plt_perc2, percentiles, i

#Waiting times
flood,cum_flood,waits = simulate_GCM_futures_probStart(Y_L15sm[2,42:],prob,N_prob=1000)


#Survival Analysis#############################################################
median = survival(waits)
AvgMedWait = np.zeros([np.shape(prob)[0]-10+1,np.shape(prob)[1]])
for i in range(np.shape(prob)[1]):
    AvgMedWait[:,i] = moving_average(median[:,i],n=10)
del i

GCM2030Waits = median[68,:]
GCM2050Waits = median[88,:]

plt.figure()
for i in range(6):
    plt.plot(GCMyears[38:89],median[38:89,i],'g',linewidth = 5)
    plt.plot(GCMyears[38:89],median[38:89,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#to 2070 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(GCMyears[38:109],median[38:109,i],'g',linewidth = 5)
    plt.plot(GCMyears[38:109],median[38:109,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)

#to 2100 - not much data supporting later years
plt.figure()
for i in range(6):
    plt.plot(GCMyears,median[:,i],'g',linewidth = 5)
    plt.plot(GCMyears,median[:,i+6],'b',linewidth = 5)
del i
plt.xlabel('Year')
plt.ylabel('Median Time Between Floods')
plt.legend(RCPs)


#Cumulative floods
plt.figure()
plt.plot(GCMyears[:],np.mean(cum_flood[:,0,:,:],axis=1),'g',linewidth = 0.2)
plt.xlabel('Year')
plt.ylabel('Cumulative Floods')
plt.title(GCMs[0])
plt.legend([RCPs[0]])
plt.ylim=[0,40]

plt.figure()
plt.plot(GCMyears[:],np.mean(cum_flood[:,6,:,:],axis=1),'b',linewidth = 0.2)
plt.xlabel('Year')
plt.ylabel('Cumulative Floods')
plt.title(GCMs[0])
plt.legend([RCPs[1]])
plt.ylim=[0,40]