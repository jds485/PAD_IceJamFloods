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
import pandas as pd

#Seaborn style
sb.set_style('whitegrid')

#Don't show plots in Spyder as they're generated
plt.ioff()

os.chdir('../bayesian_regression')

# set seed
random.seed(51321)
np.random.seed(51321)

#Define GCMs and RCPs
GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
RCPs=['RCP85','RCP45']

#Model Considering uncertainty in historical data, Y=0 for medium, small, and unknown
os.chdir('./DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_y0_1915-2020')
#Load data from R
#Prob true Z=1|X
p_L15sm_y0 = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#prob recording Y=1|X
q_L15sm_y0 = np.loadtxt('q.csv',delimiter=',',skiprows=1)
#observed data matrix 1963-2020, replicates 1915-1962 generated from q
Y_L15sm_y0 = np.loadtxt('y.csv',delimiter=',',skiprows=1)
#Replicates of Z generated from p
Zrep_L15sm_y0 = np.loadtxt('zrep.csv',delimiter=',',skiprows=1)
#observed data 1915-1962
Yobs_L15sm_y0 = np.loadtxt('yL15sm15t20.csv',delimiter=',',skiprows=1)
#years
years_L15sm = np.loadtxt('years.csv',delimiter=',',skiprows=1)
#colors for flood magnitudes
Yobs_colors = pd.read_csv('y_mag_colors.csv')
Yobs_colors = np.array(Yobs_colors['x'])

if not os.path.exists('./Historical'):
    os.mkdir('Historical')
os.chdir('Historical')

#Moving window size and percentiles to compute for plots
window=1
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
qMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
pqMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
YMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
ZrepMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])

for i in range(np.shape(p_L15sm_y0)[0]):
    YMA_L15sm_y0[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_L15sm_y0[:,i] = moving_average(Zrep_L15sm_y0[i,:],window)
    pMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:],window)
    qMA_L15sm_y0[:,i] = moving_average(q_L15sm_y0[i,:],window)
    pqMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:] - q_L15sm_y0[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF, and Recorded Data\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])],
                                     Yobs=Yobs_L15sm_y0, 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), 
                                     xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add colors for historical flood magnitudes
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), 
                                     xlim=[1915,1965], YBayes=1963, Ycolors=Yobs_colors)
plt.gca().get_legend().remove()
plt.savefig('HistPredPlot_MeanZ_MagColors_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#p and q
plt_perc2 = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Prob. (p) and Prob. Recording Large IJF (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,1970])
plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#difference plot
plt_perc = np.percentile(pqMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average [Large IJF Prob. (p) - Prob. Recording Large IJF (q)]\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability Difference',scale='linear',ylim=[-1,1],Names='p - q',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1915,1965])
plt.axhline(y = 0., color = 'blue')
plt.savefig('HistPredPlot_p-q_' + str(window) + 'yr.png', dpi = 600)
plt.close()


window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
qMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
pqMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
YMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
ZrepMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])

for i in range(np.shape(p_L15sm_y0)[0]):
    YMA_L15sm_y0[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_L15sm_y0[:,i] = moving_average(Zrep_L15sm_y0[i,:],window)
    pMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:],window)
    qMA_L15sm_y0[:,i] = moving_average(q_L15sm_y0[i,:],window)
    pqMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:] - q_L15sm_y0[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF, and Recorded Data\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])],
                                     Yobs=Yobs_L15sm_y0, 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), 
                                     xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Y and Zrep
plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#p and q
plt_perc2 = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Prob. (p) and Prob. Recording Large IJF (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,1970])
plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#difference plot
plt_perc = np.percentile(pqMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average [Large IJF Prob. (p) - Prob. Recording Large IJF (q)]\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability Difference',scale='linear',ylim=[-1,1],Names='p - q',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1915,1965])
plt.axhline(y = 0., color = 'blue')
plt.savefig('HistPredPlot_p-q_' + str(window) + 'yr.png', dpi = 600)
plt.close()

window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
qMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
pqMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
YMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])
ZrepMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])

for i in range(np.shape(p_L15sm_y0)[0]):
    YMA_L15sm_y0[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_L15sm_y0[:,i] = moving_average(Zrep_L15sm_y0[i,:],window)
    pMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:],window)
    qMA_L15sm_y0[:,i] = moving_average(q_L15sm_y0[i,:],window)
    pqMA_L15sm_y0[:,i] = moving_average(p_L15sm_y0[i,:] - q_L15sm_y0[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred Probability of Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording a Large IJF, and Recorded Data\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)],
                                     Yobs=Yobs_L15sm_y0, 
                                     xlim=[1920,1970])
plt.savefig('HistPredPlot_q_ObservedY_' + str(window) + 'yr.png', dpi = 600)
plt.close()


plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), 
                                     xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_L15sm_y0,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#p and q
plt_perc2 = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
plt_perc = np.percentile(pMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Prob. (p) and Prob. Recording Large IJF (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1920,1970])
plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#difference plot
plt_perc = np.percentile(pqMA_L15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average [Large IJF Prob. (p) - Prob. Recording Large IJF (q)]\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability Difference',scale='linear',ylim=[-1,1],Names='p - q',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1915,1965])
plt.axhline(y = 0., color = 'blue')
plt.savefig('HistPredPlot_p-q_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#GCM Plots
window=20
os.chdir('../')

#Load GCM probabilities and combine into an array
p1_y0 = np.loadtxt('GCMp_Had85.csv',delimiter=',',skiprows=1)
p2_y0 = np.loadtxt('GCMp_Acc85.csv',delimiter=',',skiprows=1)
p3_y0 = np.loadtxt('GCMp_Can85.csv',delimiter=',',skiprows=1)
p4_y0 = np.loadtxt('GCMp_CCS85.csv',delimiter=',',skiprows=1)
p5_y0 = np.loadtxt('GCMp_CNR85.csv',delimiter=',',skiprows=1)
p6_y0 = np.loadtxt('GCMp_MPI85.csv',delimiter=',',skiprows=1)
p7_y0 = np.loadtxt('GCMp_Had45.csv',delimiter=',',skiprows=1)
p8_y0 = np.loadtxt('GCMp_ACC45.csv',delimiter=',',skiprows=1)
p9_y0 = np.loadtxt('GCMp_Can45.csv',delimiter=',',skiprows=1)
p10_y0 = np.loadtxt('GCMp_CCS45.csv',delimiter=',',skiprows=1)
p11_y0 = np.loadtxt('GCMp_CNR45.csv',delimiter=',',skiprows=1)
p12_y0 = np.loadtxt('GCMp_MPI45.csv',delimiter=',',skiprows=1)
prob_y0 = np.stack([p1_y0,p2_y0,p3_y0,p4_y0,p5_y0,p6_y0,p7_y0,p8_y0,p9_y0,p10_y0,p11_y0,p12_y0], axis=2)
del p1_y0,p2_y0,p3_y0,p4_y0,p5_y0,p6_y0,p7_y0,p8_y0,p9_y0,p10_y0,p11_y0,p12_y0

GCMyears = np.loadtxt('year62t2099.csv',delimiter=',',skiprows=1)

#Re-arrange order of array index to match function
prob_y0 = np.moveaxis(prob_y0, [2,1], [1,2])

if not os.path.exists('./GCMs'):
    os.mkdir('./GCMs')
os.chdir('./GCMs')

# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(prob_y0)[0]-(window-1),np.shape(prob_y0)[2]])
    probMA2 = np.zeros([np.shape(prob_y0)[0]-(window-1),np.shape(prob_y0)[2]])
    for i in range(np.shape(prob_y0)[2]):
        probMA[:,i] = moving_average(prob_y0[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_y0[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.000001,1],Names=RCPs,
                                window=window,years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                CIind=1, xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

# matrix to store IQRs at specific years
IQRs_y0 = np.empty((len(GCMs),4*len(RCPs)))
# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(prob_y0)[0]-(window-1),np.shape(prob_y0)[2]])
    probMA2 = np.zeros([np.shape(prob_y0)[0]-(window-1),np.shape(prob_y0)[2]])
    for i in range(np.shape(prob_y0)[2]):
        probMA[:,i] = moving_average(prob_y0[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_y0[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_y0[scenario, 0] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_y0[scenario, 1] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_y0[scenario, 2] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_y0[scenario, 3] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_y0[scenario, 4] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_y0[scenario, 5] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_y0[scenario, 6] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_y0[scenario, 7] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    percentile_fill_plot_single(plt_perc,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[0],
                                window=window,CIind=0,colPlt='green', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
    percentile_fill_plot_single(plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[1],
                                window=window,CIind=0,colPlt='blue', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

#save IQRs
IQRs_y0 = pd.DataFrame(IQRs_y0)
IQRs_y0.index = GCMs
IQRs_y0.to_csv('GCM_IQRs.csv', header = ['2020-8.5','2040-8.5','2060-8.5','2080-8.5','2020-4.5','2040-4.5','2060-4.5','2080-4.5'])

#1962-present – best model based on only that data
os.chdir('../../DREAMzs_L15_VermPrecip_1962-2020')
#Load data from R
#Prob Z=1|X of flood
p_vp = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Z generated from p
Zrep_vp = np.loadtxt('zrep.csv',delimiter=',',skiprows=1)

if not os.path.exists('./Historical'):
    os.mkdir('Historical')
os.chdir('Historical')

#Moving window size and percentiles to compute for plots
window=1
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
YMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
ZrepMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])

for i in range(np.shape(p_vp)[0]):
    YMA_vp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_vp[:,i] = moving_average(Zrep_vp[i,:],window)
    pMA_vp[:,i] = moving_average(p_vp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_vp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
YMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
ZrepMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])

for i in range(np.shape(p_vp)[0]):
    YMA_vp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_vp[:,i] = moving_average(Zrep_vp[i,:],window)
    pMA_vp[:,i] = moving_average(p_vp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_vp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
YMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])
ZrepMA_vp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_vp)[0]])

for i in range(np.shape(p_vp)[0]):
    YMA_vp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_vp[:,i] = moving_average(Zrep_vp[i,:],window)
    pMA_vp[:,i] = moving_average(p_vp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_vp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_vp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

os.chdir('../')
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
#There are 2 additional years in this dataset because it is not limited by Ft. Smith
#Remove those 2 years to make the plots the same time points as other models
prob_hold = np.delete(prob_hold, [14,27], axis=0)

if not os.path.exists('./GCMs'):
    os.mkdir('./GCMs')
os.chdir('./GCMs')

window=20

# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(prob_hold)[0]-(window-1),np.shape(prob_hold)[2]])
    probMA2 = np.zeros([np.shape(prob_hold)[0]-(window-1),np.shape(prob_hold)[2]])
    for i in range(np.shape(prob_hold)[2]):
        probMA[:,i] = moving_average(prob_hold[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_hold[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.000001,1],Names=RCPs,window=window,
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                CIind=1, xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

# matrix to store IQRs at specific years
IQRs_hold = np.empty((len(GCMs),4*len(RCPs)))
# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(prob_hold)[0]-(window-1),np.shape(prob_hold)[2]])
    probMA2 = np.zeros([np.shape(prob_hold)[0]-(window-1),np.shape(prob_hold)[2]])
    for i in range(np.shape(prob_hold)[2]):
        probMA[:,i] = moving_average(prob_hold[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_hold[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_hold[scenario, 0] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_hold[scenario, 1] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_hold[scenario, 2] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_hold[scenario, 3] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_hold[scenario, 4] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_hold[scenario, 5] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_hold[scenario, 6] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_hold[scenario, 7] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    percentile_fill_plot_single(plt_perc,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[0],
                                window=window,CIind=0,colPlt='green', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
    percentile_fill_plot_single(plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Lamontagne et al. Best Model, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[1],
                                window=window,CIind=0,colPlt='blue', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

#save IQRs
IQRs_hold = pd.DataFrame(IQRs_hold)
IQRs_hold.index = GCMs
IQRs_hold.to_csv('GCM_IQRs.csv', header = ['2020-8.5','2040-8.5','2060-8.5','2080-8.5','2020-4.5','2040-4.5','2060-4.5','2080-4.5'])


#1962-present – best model with Ft. Smith
os.chdir('../../DREAMzs_L15_SmithChipVermPrecipPCA_1962-2020')
#Load data from R
#Prob Z=1|X of flood
p_cvsp = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Y generated from p
Zrep_cvsp = np.loadtxt('zrep.csv',delimiter=',',skiprows=1)

if not os.path.exists('./Historical'):
    os.mkdir('Historical')
os.chdir('Historical')

#Moving window size and percentiles to compute for plots
window=1
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
YMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
ZrepMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])

for i in range(np.shape(p_cvsp)[0]):
    YMA_cvsp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_cvsp[:,i] = moving_average(Zrep_cvsp[i,:],window)
    pMA_cvsp[:,i] = moving_average(p_cvsp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_cvsp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
YMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
ZrepMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])

for i in range(np.shape(p_cvsp)[0]):
    YMA_cvsp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_cvsp[:,i] = moving_average(Zrep_cvsp[i,:],window)
    pMA_cvsp[:,i] = moving_average(p_cvsp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_cvsp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
YMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])
ZrepMA_cvsp = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_cvsp)[0]])

for i in range(np.shape(p_cvsp)[0]):
    YMA_cvsp[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_cvsp[:,i] = moving_average(Zrep_cvsp[i,:],window)
    pMA_cvsp[:,i] = moving_average(p_cvsp[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_cvsp,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_cvsp,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

os.chdir('../')
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

if not os.path.exists('./GCMs'):
    os.mkdir('./GCMs')
os.chdir('./GCMs')

window=20

# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(prob_cvsp)[0]-(window-1),np.shape(prob_cvsp)[2]])
    probMA2 = np.zeros([np.shape(prob_cvsp)[0]-(window-1),np.shape(prob_cvsp)[2]])
    for i in range(np.shape(prob_cvsp)[2]):
        probMA[:,i] = moving_average(prob_cvsp[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_cvsp[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.000001,1],Names=RCPs,window=window,
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                CIind=1, xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

# matrix to store IQRs at specific years
IQRs_cvsp = np.empty((len(GCMs),4*len(RCPs)))
# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(prob_cvsp)[0]-(window-1),np.shape(prob_cvsp)[2]])
    probMA2 = np.zeros([np.shape(prob_cvsp)[0]-(window-1),np.shape(prob_cvsp)[2]])
    for i in range(np.shape(prob_cvsp)[2]):
        probMA[:,i] = moving_average(prob_cvsp[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_cvsp[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_cvsp[scenario, 0] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_cvsp[scenario, 1] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_cvsp[scenario, 2] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_cvsp[scenario, 3] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_cvsp[scenario, 4] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_cvsp[scenario, 5] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_cvsp[scenario, 6] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_cvsp[scenario, 7] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    percentile_fill_plot_single(plt_perc,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[0],window=window,
                                CIind=0,colPlt='green', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
    percentile_fill_plot_single(plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: Best Model with Ft. Smith, Trained 1962-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[1],
                                window=window,CIind=0,colPlt='blue', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

#save IQRs
IQRs_cvsp = pd.DataFrame(IQRs_cvsp)
IQRs_cvsp.index = GCMs
IQRs_cvsp.to_csv('GCM_IQRs.csv', header = ['2020-8.5','2040-8.5','2060-8.5','2080-8.5','2020-4.5','2040-4.5','2060-4.5','2080-4.5'])


#1915-present – no uncertainty considered
os.chdir('../../DREAMzs_L15_SmithChipVermPrecipPCA_NoUncertainty_1915-2020')
#Load data from R
#Prob Z=1|X of flood
p_NoUncertainty = np.loadtxt('p.csv',delimiter=',',skiprows=1)
#Replicates of Z generated from p
Zrep_NoUncertainty = np.loadtxt('zrep.csv',delimiter=',',skiprows=1)

if not os.path.exists('./Historical'):
    os.mkdir('Historical')
os.chdir('Historical')

#Moving window size and percentiles to compute for plots
window=1
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
YMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
ZrepMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
pqMA_NoUncL15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
qMA_L15sm_y0 = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm_y0)[0]])

for i in range(np.shape(p_NoUncertainty)[0]):
    YMA_NoUncertainty[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_NoUncertainty[:,i] = moving_average(Zrep_NoUncertainty[i,:],window)
    pMA_NoUncertainty[:,i] = moving_average(p_NoUncertainty[i,:],window)
    #To plot difference between p No Uncertainty and q with uncertainty considered
    pqMA_NoUncL15sm_y0[:,i] = moving_average(p_NoUncertainty[i,:] - q_L15sm_y0[i,:],window)
    qMA_L15sm_y0[:,i] = moving_average(q_L15sm_y0[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#p and q from uncertainty considered
plt_perc2 = np.percentile(qMA_L15sm_y0,percentiles,axis=1)
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Prob. (p) from Model with No Historical Uncertainty, and\n Prob. Recording Large IJF (q) from Model with Historical Uncertainty Considered, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1915,1962])
plt.savefig('HistPredPlot_pNoUncqUnc_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#difference plot
plt_perc = np.percentile(pqMA_NoUncL15sm_y0,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average [Large IJF Prob. (p) - Prob. Recording Large IJF (q)]\nModels: [Historical Uncertainty Not Considered - Considered], Trained 1915-2020',
                                     ylabel='Large IJF Probability Difference',scale='linear',ylim=[-1,1],Names='p - q',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1915,1962])
plt.axhline(y = 0., color = 'blue')
plt.savefig('HistPredPlot_pNoUnc-qUnc_' + str(window) + 'yr.png', dpi = 600)
plt.close()



#Moving window size and percentiles to compute for plots
window=5
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
YMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
ZrepMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])

for i in range(np.shape(p_NoUncertainty)[0]):
    YMA_NoUncertainty[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_NoUncertainty[:,i] = moving_average(Zrep_NoUncertainty[i,:],window)
    pMA_NoUncertainty[:,i] = moving_average(p_NoUncertainty[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()


#Moving window size and percentiles to compute for plots
window=10
percentiles=[2.5,25,50,75,97.5]

#Compute moving average of all variables
pMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
YMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])
ZrepMA_NoUncertainty = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_NoUncertainty)[0]])

for i in range(np.shape(p_NoUncertainty)[0]):
    YMA_NoUncertainty[:,i] = moving_average(Y_L15sm_y0[i,:],window)
    ZrepMA_NoUncertainty[:,i] = moving_average(Zrep_NoUncertainty[i,:],window)
    pMA_NoUncertainty[:,i] = moving_average(p_NoUncertainty[i,:],window)

#Compute percentiles and plot
plt_perc = np.percentile(pMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])
plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
plt.close()

plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     xlim=[1920,2020])    
#Add mean of Y
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability, Z, and Observed/Inferred Y\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.mean(Y_L15sm_y0,axis=0), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
plt.close()
#Add predicted mean of Z
percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
                                     title='Hindcast: '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm_y0,axis=0)[0:43]),np.mean(Y_L15sm_y0,axis=0)[43:]), xlim=[1920,2020], YBayes=1963)
plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
plt.close()

#Y and Zrep
plt_perc = np.percentile(ZrepMA_NoUncertainty,percentiles,axis=1)
plt_perc2 = np.percentile(YMA_NoUncertainty,percentiles,axis=1)
percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
                                     title='Hindcast: '+str(window)+'-year Average Probability of a Large IJF and Recording a Large IJF\nModel: No Historical Uncertainty, Trained 1915-2020',
                                     ylabel='Large IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted Z', 'Observed/Inferred Y'],
                                     window=window,years=years_L15sm[int(window/2-1):int(len(years_L15sm)-window/2)], 
                                     CIind=0, xlim=[1920,2020])
plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
plt.close()

os.chdir('../')
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
prob_NoUncertainty = np.stack([p1_NoUncertainty,p2_NoUncertainty,p3_NoUncertainty,p4_NoUncertainty,
                               p5_NoUncertainty,p6_NoUncertainty,p7_NoUncertainty,p8_NoUncertainty,
                               p9_NoUncertainty,p10_NoUncertainty,p11_NoUncertainty,p12_NoUncertainty], axis=2)
del p1_NoUncertainty,p2_NoUncertainty,p3_NoUncertainty,p4_NoUncertainty,p5_NoUncertainty
del p6_NoUncertainty,p7_NoUncertainty,p8_NoUncertainty,p9_NoUncertainty,p10_NoUncertainty,p11_NoUncertainty,p12_NoUncertainty

#Re-arrange order of array index to match function
prob_NoUncertainty = np.moveaxis(prob_NoUncertainty, [2,1], [1,2])

GCMyears_NoUncertainty = GCMyears

if not os.path.exists('./GCMs'):
    os.mkdir('./GCMs')
os.chdir('./GCMs')

window=20

# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [10,25,50,75,90]
    
    probMA = np.zeros([np.shape(prob_NoUncertainty)[0]-(window-1),np.shape(prob_NoUncertainty)[2]])
    probMA2 = np.zeros([np.shape(prob_NoUncertainty)[0]-(window-1),np.shape(prob_NoUncertainty)[2]])
    for i in range(np.shape(prob_NoUncertainty)[2]):
        probMA[:,i] = moving_average(prob_NoUncertainty[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_NoUncertainty[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    percentile_fill_plot_double(plt_perc,plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.000001,1],Names=RCPs,window=window,
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                CIind=1, xlim=[1980,2100])
    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

# matrix to store IQRs at specific years
IQRs_NoUncertainty = np.empty((len(GCMs),4*len(RCPs)))
# Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
for scenario in range(len(GCMs)):
    percentiles = [2.5,25,50,75,97.5]
    
    probMA = np.zeros([np.shape(prob_NoUncertainty)[0]-(window-1),np.shape(prob_NoUncertainty)[2]])
    probMA2 = np.zeros([np.shape(prob_NoUncertainty)[0]-(window-1),np.shape(prob_NoUncertainty)[2]])
    for i in range(np.shape(prob_NoUncertainty)[2]):
        probMA[:,i] = moving_average(prob_NoUncertainty[:,scenario,i],window)
        probMA2[:,i] = moving_average(prob_NoUncertainty[:,scenario+6,i],window)
    #Probability
    plt_perc = np.percentile(probMA,percentiles,axis=1)
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_NoUncertainty[scenario, 0] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_NoUncertainty[scenario, 1] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_NoUncertainty[scenario, 2] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_NoUncertainty[scenario, 3] = np.diff(plt_perc[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    
    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]]
    IQRs_NoUncertainty[scenario, 4] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2020)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]]
    IQRs_NoUncertainty[scenario, 5] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2040)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]]
    IQRs_NoUncertainty[scenario, 6] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2060)[0][0]])
    #np.average(probMA2,axis=1)[np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    #plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]]
    IQRs_NoUncertainty[scenario, 7] = np.diff(plt_perc2[[1,3],np.where(GCMyears[int(window/2-1):int(len(GCMyears)-window/2)] == 2080)[0][0]])
    
    percentile_fill_plot_single(plt_perc,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[0],
                                window=window,CIind=0,colPlt='green', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
    percentile_fill_plot_single(plt_perc2,
                                title=GCMs[scenario]+': '+str(window)+'-year Average Large IJF Probability\nModel: No Historical Uncertainty, Trained 1915-2020',
                                ylabel='Large IJF Probability',
                                scale='log',ylim=[0.00000001,1],Names=RCPs[1],
                                window=window,CIind=0,colPlt='blue', 
                                years=GCMyears[int(window/2-1):int(len(GCMyears)-window/2)], 
                                xlim=[1980,2100])
    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
    plt.close()
del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2

#save IQRs
IQRs_NoUncertainty = pd.DataFrame(IQRs_NoUncertainty)
IQRs_NoUncertainty.index = GCMs
IQRs_NoUncertainty.to_csv('GCM_IQRs.csv', header = ['2020-8.5','2040-8.5','2060-8.5','2080-8.5','2020-4.5','2040-4.5','2060-4.5','2080-4.5'])


##Waiting times
#flood,cum_flood,waits = simulate_GCM_futures_probStart(Y_L15sm_y0[2,43:],prob,N_prob=1000)
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
#for i in range(len(GCMs)):
#    plt.plot(GCMyears[38:89],median[38:89,i],'g',linewidth = 5)
#    plt.plot(GCMyears[38:89],median[38:89,i+6],'b',linewidth = 5)
#del i
#plt.xlabel('Year')
#plt.ylabel('Median Time Between Floods')
#plt.legend(RCPs)
#
##to 2070 - not much data supporting later years
#plt.figure()
#for i in range(len(GCMs)):
#    plt.plot(GCMyears[38:109],median[38:109,i],'g',linewidth = 5)
#    plt.plot(GCMyears[38:109],median[38:109,i+6],'b',linewidth = 5)
#del i
#plt.xlabel('Year')
#plt.ylabel('Median Time Between Floods')
#plt.legend(RCPs)
#
##to 2100 - not much data supporting later years
#plt.figure()
#for i in range(len(GCMs)):
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