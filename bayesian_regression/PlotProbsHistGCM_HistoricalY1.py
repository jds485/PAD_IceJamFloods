# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:13:14 2022

@author: js4yd
"""

##Model Considering uncertainty in historical data, historical Y=1 for medium, small, unknown
#os.chdir('./DREAMzs_L15_SmithChipVermPrecipPCA_Uncertainty_1915-2020')
##Load data from R
##Prob true Z=1|X
#p_L15sm = np.loadtxt('p.csv',delimiter=',',skiprows=1)
##prob recording Y=1|X
#q_L15sm = np.loadtxt('q.csv',delimiter=',',skiprows=1)
##observed data matrix 1962-2020, replicates 1915-1962 generated from q
#Y_L15sm = np.loadtxt('y.csv',delimiter=',',skiprows=1)
##Replicates of Z generated from p
#Zrep_L15sm = np.loadtxt('zrep.csv',delimiter=',',skiprows=1)
#years_L15sm = np.loadtxt('years.csv',delimiter=',',skiprows=1)
##observed data 1915-1962
#Yobs_All15sm_y0 = np.loadtxt('yAll15sm15t20.csv',delimiter=',',skiprows=1)
##colors for flood magnitudes
#Yobs_colors = pd.read_csv('y_mag_colors.csv')
#Yobs_colors = np.array(Yobs_colors['x'])
#
#if not os.path.exists('./Historical'):
#    os.mkdir('Historical')
#os.chdir('Historical')
#
##Moving window size and percentiles to compute for plots
#window=1
#percentiles=[2.5,25,50,75,97.5]
#
##Compute moving average of all variables
#pMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#qMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#YMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#ZrepMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#
#for i in range(np.shape(p_L15sm)[0]):
#    YMA_L15sm[:,i] = moving_average(Y_L15sm[i,:],window)
#    ZrepMA_L15sm[:,i] = moving_average(Zrep_L15sm[i,:],window)
#    pMA_L15sm[:,i] = moving_average(p_L15sm[i,:],window)
#    qMA_L15sm[:,i] = moving_average(q_L15sm[i,:],window)
#
##Compute percentiles and plot
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording an IJF When IJF Occurs\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
##Add mean of Y
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])],
#                                     Yobs=Yobs_All15sm_y0, 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_q_ObservedY_' + str(window) + 'yr.png', dpi = 600)
##Add predicted mean of Z
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm,axis=0)[0:42]),np.mean(Y_L15sm,axis=0)[42:]), xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
#
#
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])    
##Add mean of Y
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.mean(Y_L15sm,axis=0), 
#                                     xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
#
##Y and Zrep
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted', 'Observed/Inferred'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
#
##p and q
#plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Prob. (p) and Prob. Recording IJF | IJF Occurs (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
#
#
##Moving window size and percentiles to compute for plots
#window=5
#percentiles=[2.5,25,50,75,97.5]
#
##Compute moving average of all variables
#pMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#qMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#YMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#ZrepMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#
#for i in range(np.shape(p_L15sm)[0]):
#    YMA_L15sm[:,i] = moving_average(Y_L15sm[i,:],window)
#    ZrepMA_L15sm[:,i] = moving_average(Zrep_L15sm[i,:],window)
#    pMA_L15sm[:,i] = moving_average(p_L15sm[i,:],window)
#    qMA_L15sm[:,i] = moving_average(q_L15sm[i,:],window)
#
##Compute percentiles and plot
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording an IJF When IJF Occurs\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
##Add mean of Y
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])],
#                                     Yobs=Yobs_All15sm_y0, 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_q_ObservedY_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])    
##Add mean of Y
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.mean(Y_L15sm,axis=0), xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
##Add predicted mean of Z
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm,axis=0)[0:42]),np.mean(Y_L15sm,axis=0)[42:]), xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
#
#
##Y and Zrep
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted', 'Observed/Inferred'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
#
##p and q
#plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Prob. (p) and Prob. Recording IJF | IJF Occurs (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
#
#
##Moving window size and percentiles to compute for plots
#window=10
#percentiles=[2.5,25,50,75,97.5]
#
##Compute moving average of all variables
#pMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#qMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#YMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#ZrepMA_L15sm = np.zeros([np.shape(years_L15sm)[0]-(window-1),np.shape(p_L15sm)[0]])
#
#for i in range(np.shape(p_L15sm)[0]):
#    YMA_L15sm[:,i] = moving_average(Y_L15sm[i,:],window)
#    ZrepMA_L15sm[:,i] = moving_average(Zrep_L15sm[i,:],window)
#    pMA_L15sm[:,i] = moving_average(p_L15sm[i,:],window)
#    qMA_L15sm[:,i] = moving_average(q_L15sm[i,:],window)
#
##Compute percentiles and plot
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_p_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(qMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Probability of Recording an IJF When IJF Occurs\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_q_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average Observed/Inferred IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])
#plt.savefig('HistPredPlot_ObservedY_' + str(window) + 'yr.png', dpi = 600)
#
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     xlim=[1920,2020])    
##Add mean of Y
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.mean(Y_L15sm,axis=0), 
#                                     xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanY_' + str(window) + 'yr.png', dpi = 600)
##Add predicted mean of Z
#percentile_fill_plot_single(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))],
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names='50% and 95% CIs',
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     Yobs=np.append(np.array(np.mean(Zrep_L15sm,axis=0)[0:42]),np.mean(Y_L15sm,axis=0)[42:]), xlim=[1920,2020], YBayes=1962)
#plt.savefig('HistPredPlot_MeanZ_' + str(window) + 'yr.png', dpi = 600)
#
##Y and Zrep
#plt_perc = np.percentile(ZrepMA_L15sm,percentiles,axis=1)
#plt_perc2 = np.percentile(YMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['Predicted', 'Observed/Inferred'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_PPC_' + str(window) + 'yr.png', dpi = 600)
#
##p and q
#plt_perc2 = np.percentile(qMA_L15sm,percentiles,axis=1)
#plt_perc = np.percentile(pMA_L15sm,percentiles,axis=1)
#percentile_fill_plot_double(plt_perc[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     plt_perc2[:,0:(np.shape(years_L15sm)[0]-(window-1))], 
#                                     title='Hindcast: '+str(window)+'-year Average IJF Prob. (p) and Prob. Recording IJF | IJF Occurs (q)\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                     ylabel='IJF Probability',scale='linear',ylim=[0,1],Names=['p', 'q'],
#                                     window=window,years=years_L15sm[(window-1):(np.shape(years_L15sm)[0])], 
#                                     CIind=0, xlim=[1920,2020])
#plt.savefig('HistPredPlot_pq_' + str(window) + 'yr.png', dpi = 600)
#
#
##GCM Plots
#GCMs=['HadGEM2-ES','ACCESS1-0','CanESM2','CCSM4','CNRM-CM5','MPI-ESM-LR']
#RCPs=['RCP85','RCP45']
#window=20
#os.chdir('../')
#
##Load GCM probabilities and combine into an array
#p1 = np.loadtxt('GCMp_Had85.csv',delimiter=',',skiprows=1)
#p2 = np.loadtxt('GCMp_Acc85.csv',delimiter=',',skiprows=1)
#p3 = np.loadtxt('GCMp_Can85.csv',delimiter=',',skiprows=1)
#p4 = np.loadtxt('GCMp_CCS85.csv',delimiter=',',skiprows=1)
#p5 = np.loadtxt('GCMp_CNR85.csv',delimiter=',',skiprows=1)
#p6 = np.loadtxt('GCMp_MPI85.csv',delimiter=',',skiprows=1)
#p7 = np.loadtxt('GCMp_Had45.csv',delimiter=',',skiprows=1)
#p8 = np.loadtxt('GCMp_ACC45.csv',delimiter=',',skiprows=1)
#p9 = np.loadtxt('GCMp_Can45.csv',delimiter=',',skiprows=1)
#p10 = np.loadtxt('GCMp_CCS45.csv',delimiter=',',skiprows=1)
#p11 = np.loadtxt('GCMp_CNR45.csv',delimiter=',',skiprows=1)
#p12 = np.loadtxt('GCMp_MPI45.csv',delimiter=',',skiprows=1)
#prob = np.stack([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], axis=2)
#del p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12
#
##Re-arrange order of array index to match function
#prob = np.moveaxis(prob, [2,1], [1,2])
#
#GCMyears = np.loadtxt('year62t2099.csv',delimiter=',',skiprows=1)
#
#if not os.path.exists('./GCMs'):
#    os.mkdir('./GCMs')
#os.chdir('./GCMs')
#
## Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
#for scenario in range(len(GCMs)):
#    percentiles = [10,25,50,75,90]
#    
#    probMA = np.zeros([np.shape(prob)[0]-(window-1),np.shape(prob)[2]])
#    probMA2 = np.zeros([np.shape(prob)[0]-(window-1),np.shape(prob)[2]])
#    for i in range(np.shape(prob)[2]):
#        probMA[:,i] = moving_average(prob[:,scenario,i],window)
#        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
#    #Probability
#    plt_perc = np.percentile(probMA,percentiles,axis=1)
#    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
#    percentile_fill_plot_double(plt_perc,plt_perc2,
#                                title=GCMs[scenario]+': '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                ylabel='IJF Probability',
#                                scale='log',ylim=[0.000001,1],Names=RCPs,
#                                window=window,years=GCMyears[(window-1):(np.shape(GCMyears)[0])], 
#                                CIind=1, xlim=[1980,2100])
#    plt.savefig('ProjPredPlot_' + GCMs[scenario] + '_' + str(window) + 'yr.png', dpi = 600)
#del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2
#
## Moving Average in Real Space with a plot in log scale, correct CI on 20-yr average
#for scenario in range(len(GCMs)):
#    percentiles = [2.5,25,50,75,97.5]
#    
#    probMA = np.zeros([np.shape(prob)[0]-(window-1),np.shape(prob)[2]])
#    probMA2 = np.zeros([np.shape(prob)[0]-(window-1),np.shape(prob)[2]])
#    for i in range(np.shape(prob)[2]):
#        probMA[:,i] = moving_average(prob[:,scenario,i],window)
#        probMA2[:,i] = moving_average(prob[:,scenario+6,i],window)
#    #Probability
#    plt_perc = np.percentile(probMA,percentiles,axis=1)
#    #np.average(probMA,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]]
#    #plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]]
#    np.diff(plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]])
#    #np.average(probMA,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]]
#    #plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]]
#    np.diff(plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]])
#    #np.average(probMA,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]]
#    #plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]]
#    np.diff(plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]])
#    #np.average(probMA,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]]
#    #plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]]
#    np.diff(plt_perc[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]])
#    
#    plt_perc2 = np.percentile(probMA2,percentiles,axis=1)
#    #np.average(probMA2,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]]
#    #plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]]
#    np.diff(plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2020)[0][0]])
#    #np.average(probMA2,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]]
#    #plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]]
#    np.diff(plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2040)[0][0]])
#    #np.average(probMA2,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]]
#    #plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]]
#    np.diff(plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2060)[0][0]])
#    #np.average(probMA2,axis=1)[np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]]
#    #plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]]
#    np.diff(plt_perc2[[1,3],np.where(GCMyears[(window-1):(np.shape(GCMyears)[0])] == 2080)[0][0]])
#    
#    percentile_fill_plot_single(plt_perc,
#                                title=GCMs[scenario]+': '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                ylabel='IJF Probability',
#                                scale='log',ylim=[0.00000001,1],Names=RCPs[0],
#                                window=window,CIind=0,colPlt='green', 
#                                years=GCMyears[(window-1):(np.shape(GCMyears)[0])], xlim=[1980,2100])
#    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[0] + '_' + str(window) + 'yr.png', dpi = 600)
#    percentile_fill_plot_single(plt_perc2,
#                                title=GCMs[scenario]+': '+str(window)+'-year Average IJF Probability\nModel: Historical Uncertainty Considered, Trained 1915-2020',
#                                ylabel='IJF Probability',
#                                scale='log',ylim=[0.00000001,1],Names=RCPs[1],
#                                window=window,CIind=0,colPlt='blue', 
#                                years=GCMyears[(window-1):(np.shape(GCMyears)[0])], xlim=[1980,2100])
#    plt.savefig('ProjPredPlot_CIs_' + GCMs[scenario] + '_' + RCPs[1] + '_' + str(window) + 'yr.png', dpi = 600)
#del scenario, plt_perc, plt_perc2, percentiles, i, probMA, probMA2