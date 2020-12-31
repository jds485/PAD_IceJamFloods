# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 07:35:26 2020

@author: jlamon02
"""
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter

#moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def percentile_fill_plot_double(Y,Y2,title='Wicked Pissah',ylabel='Cumulative Pissah',scale='linear',Names=['Pissah1','Pissah2'],ylim=None,xlim=None):
#    import matplotlib.pyplot as plt
    N = np.shape(Y)[0]
    half = int((N-1)/2)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
    #ax1.plot(np.arange(1946,2100,1),Y[half,:],color='k')
    for i in range(1,half):
        #ax1.fill_between(np.arange(0,155,1), plt_perc[i,:],plt_perc[-(i+1),:],color=colormap(i/half))
        ax1.fill_between(np.arange(1962,2100,1), Y[i,:],Y[-(i+1),:],color="green",alpha=0.5,label=Names[0])#blue

    #ax1.plot(np.arange(1946,2100,1),Y2[half,:],color='k')
    for i in range(1,half):
        #ax1.fill_between(np.arange(0,155,1), plt_perc[i,:],plt_perc[-(i+1),:],color=colormap(i/half))
        ax1.fill_between(np.arange(1962,2100,1), Y2[i,:],Y2[-(i+1),:],color="blue",alpha=0.5,label=Names[1])#red
    ax1.legend()
    ax1.set_yscale(scale)
    ax1.set_title(title, fontsize=15)
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    if ylim != None:
        plt.ylim(ylim)
    if xlim != None:
        plt.xlim(xlim)
    fig.tight_layout()

def percentile_plot_single(Y,title='Wicked Pissah',ylabel='Cumulative Pissah',scale='linear',ylim=None,xlim=None,split='scenario',Names=['Pissah1','Pissah2']):
    import matplotlib.pyplot as plt

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
    if split == 'scenario':
        ax1.plot(np.arange(1962,2100,1),Y[0,:],'g',linewidth=5,label=Names[0])
        ax1.plot(np.arange(1962,2100,1),Y[0,:],'b',linewidth=5,label=Names[1])
        ax1.legend()
        for i in range(6):
            ax1.plot(np.arange(1962,2100,1),Y[i,:],'g',linewidth=5,label=Names[0])
        for j in range(6,12):
            ax1.plot(np.arange(1962,2100,1),Y[j,:],'b',linewidth=5,label=Names[1])
    elif split == 'gcm':
        colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
        for i in range(6):
            ax1.plot(np.arange(1962,2100,1),Y[i,:],color=colors[i],linewidth=5,label=Names[i])
        ax1.legend()
        for i in range(6):
            ax1.plot(np.arange(1962,2100,1),Y[i,:],color=colors[i],linewidth=5,label=Names[i])
            ax1.plot(np.arange(1962,2100,1),Y[i+6,:],color=colors[i],linewidth=5,label=Names[i])
    
    ax1.set_yscale(scale)
    ax1.set_title(title, fontsize=15)
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    if ylim != None:
        plt.ylim(ylim)
    if xlim != None:
        plt.xlim(xlim)
    fig.tight_layout()

def survival(waits):
    N_years,N_scen,N_prob,M_boot = np.shape(waits)
    median = np.zeros([N_years,N_scen])
    for year in range(N_years):
        for GCM in range(N_scen):
            #First re-structure as a long-vector
            wait_hold = np.copy(np.reshape(waits[year,GCM,:,:],[N_prob*M_boot]))
            if np.median(wait_hold)<300:
                median[year,GCM] = np.median(wait_hold)
            else:
                E = wait_hold < 300.
                wait_hold[wait_hold>300.] = N_years-year
                
                wf = WeibullFitter().fit(wait_hold,E)
                median[year,GCM] = wf.median_survival_time_
                print('survival',GCM)
        print(N_years-year+1)
    return median