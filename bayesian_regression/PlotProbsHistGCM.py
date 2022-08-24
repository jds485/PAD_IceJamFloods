# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:32:51 2021

@author: js4yd

run in PAD_IceJamFloods\bayesian_regression directory
"""

import os
os.chdir('../logistic_regression')

from utils import *
from utils_figures import *
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb

os.chdir('../bayesian_regression')

# set seed
random.seed(51321)
np.random.seed(51321)

#Model Considering uncertainty in historical data
os.chdir('./DREAMzs2_L15_FixY_PCAcvs_pSensSpec_Historical1s')
#Load data from R
#Prob Z=1|X of flood
p_L15sm = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#prob Y=1|X of flood
q_L15sm = np.loadtxt('q.csv',delimiter=',',skiprows=1)
#observed data matrix 1915-1962 generated from q
Y_L15sm = np.loadtxt('y.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Yrep_L15sm = np.loadtxt('yrep.csv',delimiter=',',skiprows=1)
years_L15sm = np.loadtxt('years.csv',delimiter=',',skiprows=1)

#Moving window size and percentiles to compute for plots
window=5
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
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average Probability of Recording IJF for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average Observed IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)

#p and q
plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)


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
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average Probability of Recording IJF for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average Observed IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_L15sm,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)

#p and q
plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)


#GCM Plots
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
window=20

#Load GCM probabilities and combine into an array
p1 = np.loadtxt('GCMp_Had85.csv',delimiter=',',skiprows=1)
p2 = np.loadtxt('GCMp_Acc85.csv',delimiter=',',skiprows=1)
p3 = np.loadtxt('GCMp_Can85.csv',delimiter=',',skiprows=1)
p4 = np.loadtxt('GCMp_CCS85.csv',delimiter=',',skiprows=1)
p5 = np.loadtxt('GCMp_CNR85.csv',delimiter=',',skiprows=1)
p6 = np.loadtxt('GCMp_MPI85.csv',delimiter=',',skiprows=1)
p7 = np.loadtxt('GCMp_Had45.csv',delimiter=',',skiprows=1)
p8 = np.loadtxt('GCMp_ACC45.csv',delimiter=',',skiprows=1)
p9 = np.loadtxt('GCMp_Can45.csv',delimiter=',',skiprows=1)
p10 = np.loadtxt('GCMp_CCS45.csv',delimiter=',',skiprows=1)
p11 = np.loadtxt('GCMp_CNR45.csv',delimiter=',',skiprows=1)
p12 = np.loadtxt('GCMp_MPI45.csv',delimiter=',',skiprows=1)
prob = np.stack([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], axis=2)
del p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12

#Re-arrange order of array index to match function
prob = np.moveaxis(prob, [2,1], [1,2])

GCMyears = np.loadtxt('year62t2099.csv',delimiter=',',skiprows=1)

sb.set_style('darkgrid')

if not os.path.exists('./GCMs'):
    os.mkdir('GCMs')
os.chdir('GCMs')

# Moving Average in Real Space with a plot in log scale
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.000001,1],Names=RCPs,window=window,years=GCMyears[(window-1):(np.shape(GCMyears)[0])], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i
# Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green',years=GCMyears[(window-1):(np.shape(GCMyears)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue',years=GCMyears[(window-1):(np.shape(GCMyears)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i

##Waiting times
#flood,cum_flood,waits = simulate_GCM_futures_probStart(Y_L15sm[2,42:],prob,N_prob=1000)
#
#
##Survival Analysis#############################################################
#median = survival(waits)
#AvgMedWait = np.zeros([np.shape(prob)[0]-10+1,np.shape(prob)[1]])
#for i in range(np.shape(prob)[1]):
#    AvgMedWait[:,i] = moving_average(median[:,i],n=10)
#del i
#
#GCM2030Waits = median[68,:]
#GCM2050Waits = median[88,:]
#
#plt.figure()
#for i in range(6):
#    plt.plot(GCMyears[38:89],median[38:89,i],'g',linewidth = 5)
#    plt.plot(GCMyears[38:89],median[38:89,i+6],'b',linewidth = 5)
#del i
#plt.xlabel('Year')
#plt.ylabel('Median Time Between Floods')
#plt.legend(RCPs)
#
##to 2070 - not much data supporting later years
#plt.figure()
#for i in range(6):
#    plt.plot(GCMyears[38:109],median[38:109,i],'g',linewidth = 5)
#    plt.plot(GCMyears[38:109],median[38:109,i+6],'b',linewidth = 5)
#del i
#plt.xlabel('Year')
#plt.ylabel('Median Time Between Floods')
#plt.legend(RCPs)
#
##to 2100 - not much data supporting later years
#plt.figure()
#for i in range(6):
#    plt.plot(GCMyears,median[:,i],'g',linewidth = 5)
#    plt.plot(GCMyears,median[:,i+6],'b',linewidth = 5)
#del i
#plt.xlabel('Year')
#plt.ylabel('Median Time Between Floods')
#plt.legend(RCPs)
#
#
##Cumulative floods
#plt.figure()
#plt.plot(GCMyears[:],np.mean(cum_flood[:,0,:,:],axis=1),'g',linewidth = 0.2)
#plt.xlabel('Year')
#plt.ylabel('Cumulative Floods')
#plt.title(GCMs[0])
#plt.legend([RCPs[0]])
#plt.ylim=[0,40]
#
#plt.figure()
#plt.plot(GCMyears[:],np.mean(cum_flood[:,6,:,:],axis=1),'b',linewidth = 0.2)
#plt.xlabel('Year')
#plt.ylabel('Cumulative Floods')
#plt.title(GCMs[0])
#plt.legend([RCPs[1]])
#plt.ylim=[0,40]


#1962-present – best model based on only that data
os.chdir('../../DREAMzs1p')
sb.set_style('white')
#Load data from R
#Prob Z=1|X of flood
p_vp = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Yrep_vp = np.loadtxt('yrep.csv',delimiter=',',skiprows=1)

#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_vp[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_vp[:,i] = moving_average(Yrep_vp[i,:],window)
    pMA_vp[:,i] = moving_average(p_vp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_vp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_vp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_vp[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_vp[:,i] = moving_average(Yrep_vp[i,:],window)
    pMA_vp[:,i] = moving_average(p_vp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_vp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_vp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Load GCM probabilities and combine into an array
p1_hold = np.loadtxt('GCMp_Had85_hold.csv',delimiter=',',skiprows=1)
p2_hold = np.loadtxt('GCMp_Acc85_hold.csv',delimiter=',',skiprows=1)
p3_hold = np.loadtxt('GCMp_Can85_hold.csv',delimiter=',',skiprows=1)
p4_hold = np.loadtxt('GCMp_CCS85_hold.csv',delimiter=',',skiprows=1)
p5_hold = np.loadtxt('GCMp_CNR85_hold.csv',delimiter=',',skiprows=1)
p6_hold = np.loadtxt('GCMp_MPI85_hold.csv',delimiter=',',skiprows=1)
p7_hold = np.loadtxt('GCMp_Had45_hold.csv',delimiter=',',skiprows=1)
p8_hold = np.loadtxt('GCMp_ACC45_hold.csv',delimiter=',',skiprows=1)
p9_hold = np.loadtxt('GCMp_Can45_hold.csv',delimiter=',',skiprows=1)
p10_hold = np.loadtxt('GCMp_CCS45_hold.csv',delimiter=',',skiprows=1)
p11_hold = np.loadtxt('GCMp_CNR45_hold.csv',delimiter=',',skiprows=1)
p12_hold = np.loadtxt('GCMp_MPI45_hold.csv',delimiter=',',skiprows=1)
prob_hold = np.stack([p1_hold,p2_hold,p3_hold,p4_hold,p5_hold,p6_hold,p7_hold,p8_hold,p9_hold,p10_hold,p11_hold,p12_hold], axis=2)
del p1_hold,p2_hold,p3_hold,p4_hold,p5_hold,p6_hold,p7_hold,p8_hold,p9_hold,p10_hold,p11_hold,p12_hold

#Re-arrange order of array index to match function
prob_hold = np.moveaxis(prob_hold, [2,1], [1,2])

GCMyears_hold = np.loadtxt('year62t2099.csv',delimiter=',',skiprows=1)

sb.set_style('darkgrid')

if not os.path.exists('./GCMs'):
    os.mkdir('GCMs')
os.chdir('GCMs')

# PAPER = Moving Average in Real Space with a plot in log scale
window=20
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob_hold[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_hold[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_hold)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_hold)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob_hold)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob_hold)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.000001,1],Names=RCPs,window=window,years=GCMyears_hold[(window-1):(np.shape(GCMyears_hold)[0])], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob_hold[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_hold[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_hold)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_hold)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob_hold)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green',years=GCMyears_hold[(window-1):(np.shape(GCMyears_hold)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob_hold)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue',years=GCMyears_hold[(window-1):(np.shape(GCMyears_hold)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i


#1962-present – best model with Ft. Smith
os.chdir('../../DREAMzs3p_AGU')
sb.set_style('white')
#Load data from R
#Prob Z=1|X of flood
p_cvsp = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Yrep_cvsp = np.loadtxt('yrep.csv',delimiter=',',skiprows=1)

#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_cvsp[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_cvsp[:,i] = moving_average(Yrep_cvsp[i,:],window)
    pMA_cvsp[:,i] = moving_average(p_cvsp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_cvsp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_cvsp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_cvsp[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_cvsp[:,i] = moving_average(Yrep_cvsp[i,:],window)
    pMA_cvsp[:,i] = moving_average(p_cvsp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_cvsp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_cvsp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Load GCM probabilities and combine into an array
p1_cvsp = np.loadtxt('GCMp_Had85_cvsp.csv',delimiter=',',skiprows=1)
p2_cvsp = np.loadtxt('GCMp_Acc85_cvsp.csv',delimiter=',',skiprows=1)
p3_cvsp = np.loadtxt('GCMp_Can85_cvsp.csv',delimiter=',',skiprows=1)
p4_cvsp = np.loadtxt('GCMp_CCS85_cvsp.csv',delimiter=',',skiprows=1)
p5_cvsp = np.loadtxt('GCMp_CNR85_cvsp.csv',delimiter=',',skiprows=1)
p6_cvsp = np.loadtxt('GCMp_MPI85_cvsp.csv',delimiter=',',skiprows=1)
p7_cvsp = np.loadtxt('GCMp_Had45_cvsp.csv',delimiter=',',skiprows=1)
p8_cvsp = np.loadtxt('GCMp_ACC45_cvsp.csv',delimiter=',',skiprows=1)
p9_cvsp = np.loadtxt('GCMp_Can45_cvsp.csv',delimiter=',',skiprows=1)
p10_cvsp = np.loadtxt('GCMp_CCS45_cvsp.csv',delimiter=',',skiprows=1)
p11_cvsp = np.loadtxt('GCMp_CNR45_cvsp.csv',delimiter=',',skiprows=1)
p12_cvsp = np.loadtxt('GCMp_MPI45_cvsp.csv',delimiter=',',skiprows=1)
prob_cvsp = np.stack([p1_cvsp,p2_cvsp,p3_cvsp,p4_cvsp,p5_cvsp,p6_cvsp,p7_cvsp,p8_cvsp,p9_cvsp,p10_cvsp,p11_cvsp,p12_cvsp], axis=2)
del p1_cvsp,p2_cvsp,p3_cvsp,p4_cvsp,p5_cvsp,p6_cvsp,p7_cvsp,p8_cvsp,p9_cvsp,p10_cvsp,p11_cvsp,p12_cvsp

#Re-arrange order of array index to match function
prob_cvsp = np.moveaxis(prob_cvsp, [2,1], [1,2])

GCMyears_cvsp = GCMyears

sb.set_style('darkgrid')

if not os.path.exists('./GCMs'):
    os.mkdir('GCMs')
os.chdir('GCMs')

# PAPER = Moving Average in Real Space with a plot in log scale
window=20
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob_cvsp[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_cvsp[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_cvsp)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_cvsp)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob_cvsp)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob_cvsp)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.000001,1],Names=RCPs,window=window,years=GCMyears_cvsp[(window-1):(np.shape(GCMyears_cvsp)[0])], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob_cvsp[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_cvsp[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_cvsp)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_cvsp)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob_cvsp)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green',years=GCMyears_cvsp[(window-1):(np.shape(GCMyears_cvsp)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob_cvsp)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue',years=GCMyears_cvsp[(window-1):(np.shape(GCMyears_cvsp)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i



#1915-present – no uncertainty considered
os.chdir('../../DREAMzs_L15_NoUncertainty')
sb.set_style('white')
#Load data from R
#Prob Z=1|X of flood
p_NoUncertainty = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Yrep_NoUncertainty = np.loadtxt('yrep.csv',delimiter=',',skiprows=1)

#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_NoUncertainty[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_NoUncertainty[:,i] = moving_average(Yrep_NoUncertainty[i,:],window)
    pMA_NoUncertainty[:,i] = moving_average(p_NoUncertainty[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_NoUncertainty,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])
YrepMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),1001])

for i in range(1001):
    YMA_NoUncertainty[:,i] = moving_average(Y_L15sm[i,:],window)
    YrepMA_NoUncertainty[:,i] = moving_average(Yrep_NoUncertainty[i,:],window)
    pMA_NoUncertainty[:,i] = moving_average(p_NoUncertainty[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)

plt_perc = np.percentile(YrepMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], Yobs=np.mean(Y_L15sm,axis=0))
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)

#Y and Yrep
plt_perc = np.percentile(YrepMA_NoUncertainty,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], title=str(window)+'-year Average IJF Probability for Historical Record',ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Yrep', 'Y'],window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], CIind=0)
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)


#Load GCM probabilities and combine into an array
p1_NoUncertainty = np.loadtxt('GCMp_Had85_NoUncertainty.csv',delimiter=',',skiprows=1)
p2_NoUncertainty = np.loadtxt('GCMp_Acc85_NoUncertainty.csv',delimiter=',',skiprows=1)
p3_NoUncertainty = np.loadtxt('GCMp_Can85_NoUncertainty.csv',delimiter=',',skiprows=1)
p4_NoUncertainty = np.loadtxt('GCMp_CCS85_NoUncertainty.csv',delimiter=',',skiprows=1)
p5_NoUncertainty = np.loadtxt('GCMp_CNR85_NoUncertainty.csv',delimiter=',',skiprows=1)
p6_NoUncertainty = np.loadtxt('GCMp_MPI85_NoUncertainty.csv',delimiter=',',skiprows=1)
p7_NoUncertainty = np.loadtxt('GCMp_Had45_NoUncertainty.csv',delimiter=',',skiprows=1)
p8_NoUncertainty = np.loadtxt('GCMp_ACC45_NoUncertainty.csv',delimiter=',',skiprows=1)
p9_NoUncertainty = np.loadtxt('GCMp_Can45_NoUncertainty.csv',delimiter=',',skiprows=1)
p10_NoUncertainty = np.loadtxt('GCMp_CCS45_NoUncertainty.csv',delimiter=',',skiprows=1)
p11_NoUncertainty = np.loadtxt('GCMp_CNR45_NoUncertainty.csv',delimiter=',',skiprows=1)
p12_NoUncertainty = np.loadtxt('GCMp_MPI45_NoUncertainty.csv',delimiter=',',skiprows=1)
prob_NoUncertainty = np.stack([p1_NoUncertainty,p2_NoUncertainty,p3_NoUncertainty,p4_NoUncertainty,p5_NoUncertainty,p6_NoUncertainty,p7_NoUncertainty,p8_NoUncertainty,p9_NoUncertainty,p10_NoUncertainty,p11_NoUncertainty,p12_NoUncertainty], axis=2)
del p1_NoUncertainty,p2_NoUncertainty,p3_NoUncertainty,p4_NoUncertainty,p5_NoUncertainty,p6_NoUncertainty,p7_NoUncertainty,p8_NoUncertainty,p9_NoUncertainty,p10_NoUncertainty,p11_NoUncertainty,p12_NoUncertainty

#Re-arrange order of array index to match function
prob_NoUncertainty = np.moveaxis(prob_NoUncertainty, [2,1], [1,2])

GCMyears_NoUncertainty = GCMyears

sb.set_style('darkgrid')

if not os.path.exists('./GCMs'):
    os.mkdir('GCMs')
os.chdir('GCMs')

# PAPER = Moving Average in Real Space with a plot in log scale
window=20
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob_NoUncertainty[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_NoUncertainty[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_NoUncertainty)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_NoUncertainty)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob_NoUncertainty)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob_NoUncertainty)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.000001,1],Names=RCPs,window=window,years=GCMyears_NoUncertainty[(window-1):(np.shape(GCMyears_NoUncertainty)[0])], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob_NoUncertainty[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_NoUncertainty[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_NoUncertainty)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_NoUncertainty)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob_NoUncertainty)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green',years=GCMyears_NoUncertainty[(window-1):(np.shape(GCMyears_NoUncertainty)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob_NoUncertainty)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue',years=GCMyears_NoUncertainty[(window-1):(np.shape(GCMyears_NoUncertainty)[0])])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
del scenario, plt_perc, plt_perc2, percentiles, i



#Compare the Bayesian beta distribution to the logistic parametric bootstrap

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

#Load beta_boot with Bayesian estimated betas
beta_boot_Bayes = np.loadtxt('../bayesian_regression/DREAMzs1p/BayesBetas_hold.csv',delimiter=',',skiprows=1)
beta_boot_BayesLg = np.loadtxt('../bayesian_regression/DREAMzs1p/BayesBetasLg_hold.csv',delimiter=',',skiprows=1)

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
Y_HistPred = np.exp(res_Firth.params[0] + X_All[:,2] * res_Firth.params[1] + X_All[:,4] * res_Firth.params[2])/(1+np.exp(res_Firth.params[0] + X_All[:,2] * res_Firth.params[1] + X_All[:,4] * res_Firth.params[2]))
Y_HistBootPred = np.zeros([len(X_All[:,0]),len(betas_boot_Bayes_df.iloc[:,0])])
for i in range(len(betas_boot_Bayes_df.iloc[:,0])):
    Y_HistBootPred[:,i] = np.exp(np.array(betas_boot_Bayes_df.iloc[i,0]) + X_All[:,2] * np.array(betas_boot_Bayes_df.iloc[i,1]) + X_All[:,4] * np.array(betas_boot_Bayes_df.iloc[i,2]))/(1+np.exp(np.array(betas_boot_Bayes_df.iloc[i,0]) + X_All[:,2] * np.array(betas_boot_Bayes_df.iloc[i,1]) + X_All[:,4] * np.array(betas_boot_Bayes_df.iloc[i,2])))
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
prob_B,flood_B,cum_flood_B,waits_B = simulate_GCM_futures(Y_GCM,bootstrap_X[:,[1,3],:],bootstrap_Y,beta_boot,Temp_GCM,Precip_GCM,M_boot=1000,N_prob=1000)

GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']
window = 20

# PAPER = Moving Average in Real Space with a plot in log scale
for scenario in range(6):
    percentiles = [10,25,50,75,90]
    #Probability
    plt_perc = np.percentile(prob_B[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_B[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_B)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_B)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_double(plt_perc[:,0:(np.shape(prob_B)[0]-(window-1))],plt_perc2[:,0:(np.shape(prob_B)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.0001,1],Names=RCPs,window=window, xlim=[1980,2100])
del scenario, plt_perc, plt_perc2, percentiles, i
# PAPER = Moving Average in Real Space with a plot in log scale with IQR and 95% corridors
for scenario in range(6):
    percentiles = [2.5,25,50,75,97.5]
    #Probability
    plt_perc = np.percentile(prob_B[:,scenario,:],percentiles,axis=1)
    plt_perc2 = np.percentile(prob_B[:,scenario+6,:],percentiles,axis=1)

    for i in range(5):
        plt_perc[i,0:(np.shape(prob_B)[0]-(window-1))]=moving_average(plt_perc[i,:],window)
        plt_perc2[i,0:(np.shape(prob_B)[0]-(window-1))]=moving_average(plt_perc2[i,:],window)

    percentile_fill_plot_single(plt_perc[:,0:(np.shape(prob_B)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,CIind=0,colPlt='green')
    percentile_fill_plot_single(plt_perc2[:,0:(np.shape(prob_B)[0]-(window-1))],title=GCMs[scenario],ylabel='IJF Probability',scale='log',ylim=[0.00000001,1],Names=RCPs[1],window=window,CIind=0,colPlt='blue')
del scenario, plt_perc, plt_perc2, percentiles, i