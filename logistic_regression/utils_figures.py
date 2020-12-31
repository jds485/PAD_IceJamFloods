# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 07:35:26 2020

@author: jlamon02
"""
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter

from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice

#moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[(n - 1):] / n

#From https://code.activestate.com/recipes/578480-running-median-mean-and-mode/
def RunningMedian(seq, M):
    """
     Purpose: Find the median for the points in a sliding window (odd number in size) 
              as it is moved from left to right by one point at a time.
      Inputs:
            seq -- list containing items for which a running median (in a sliding window) 
                   is to be calculated
              M -- number of items in window (window size) -- must be an integer > 1
      Otputs:
         medians -- list of medians with size N - M + 1
       Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order. 
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """   
    seq = iter(seq)
    s = []   
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]    
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes    
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()    
    medians = [median()]   

    # Now slide the window by one point to the right for each new position (each pass through 
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old 
        insort(s, item)            # insert newest such that new sort is not required        
        medians.append(median())  
    return medians

def percentile_fill_plot_double(Y,Y2,title='Wicked Pissah',ylabel='Cumulative Pissah',scale='linear',Names=['Pissah1','Pissah2'],ylim=None,xlim=None,window=20):
#    import matplotlib.pyplot as plt
    N = np.shape(Y)[0]
    half = int((N-1)/2)
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
    #ax1.plot(np.arange(1946,2100,1),Y[half,:],color='k')
    for i in range(1,half):
        #ax1.fill_between(np.arange(0,155,1), plt_perc[i,:],plt_perc[-(i+1),:],color=colormap(i/half))
        ax1.fill_between(np.arange(1962+(window-1),2100,1), Y[i,:],Y[-(i+1),:],color="green",alpha=0.5,label=Names[0])#blue

    #ax1.plot(np.arange(1946,2100,1),Y2[half,:],color='k')
    for i in range(1,half):
        #ax1.fill_between(np.arange(0,155,1), plt_perc[i,:],plt_perc[-(i+1),:],color=colormap(i/half))
        ax1.fill_between(np.arange(1962+(window-1),2100,1), Y2[i,:],Y2[-(i+1),:],color="blue",alpha=0.5,label=Names[1])#red
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

def percentile_plot_single(Y,title='Wicked Pissah',ylabel='Cumulative Pissah',scale='linear',ylim=None,xlim=None,split='scenario',Names=['Pissah1','Pissah2'],window=20):
    import matplotlib.pyplot as plt

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
    if split == 'scenario':
        ax1.plot(np.arange(1962+(window-1),2100,1),Y[0,:],'g',linewidth=5,label=Names[0])
        ax1.plot(np.arange(1962+(window-1),2100,1),Y[0,:],'b',linewidth=5,label=Names[1])
        ax1.legend()
        for i in range(6):
            ax1.plot(np.arange(1962+(window-1),2100,1),Y[i,:],'g',linewidth=5,label=Names[0])
        for j in range(6,12):
            ax1.plot(np.arange(1962+(window-1),2100,1),Y[j,:],'b',linewidth=5,label=Names[1])
    elif split == 'gcm':
        colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
        for i in range(6):
            ax1.plot(np.arange(1962+(window-1),2100,1),Y[i,:],color=colors[i],linewidth=5,label=Names[i])
        ax1.legend()
        for i in range(6):
            ax1.plot(np.arange(1962+(window-1),2100,1),Y[i,:],color=colors[i],linewidth=5,label=Names[i])
            ax1.plot(np.arange(1962+(window-1),2100,1),Y[i+6,:],color=colors[i],linewidth=5,label=Names[i])
    
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