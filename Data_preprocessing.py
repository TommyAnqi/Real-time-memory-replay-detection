# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import pandas as pd
import random
import numba
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# transform the original data to binary spiking sequences by different time bin size. The range of time bin size is from 1-1000 milliseconds.
def data_pre(binsize_sleep, binsize_awake, binsize_psleep, rat_name, record_name,position_data_name, epochs, original_data, position): 
    
    # extract the start time and end time for presleep, awake and post-sleep phases
    pre_start = epochs[record_name][0] * 1000
    pre_end = epochs[record_name][1] * 1000
    awake_start = epochs[record_name][2] * 1000
    awake_end = epochs[record_name][3] * 1000
    sleep_start = epochs[record_name][4] * 1000
    sleep_end = epochs[record_name][5] * 1000
    
    # extract all the neurons in hippocampus and compute how many time steps in particular time bin size
    x = 0
    idx = original_data[rat_name]['area'] == 'hc'    
    
    pre_data0 = np.arange((pre_end - pre_start) / binsize_sleep)
    df_pre_allneurons = pd.DataFrame(pre_data0, index = pre_data0)
    df_pre_allneurons_count = pd.DataFrame(pre_data0, index = pre_data0)
    
    awake_data0 = np.arange((awake_end-awake_start) / binsize_awake)
    df_awake_allneurons = pd.DataFrame(awake_data0, index = awake_data0)
    df_awake_allneurons_count = pd.DataFrame(awake_data0, index = awake_data0)
    
    sleep_data0 = np.arange((sleep_end-sleep_start) / binsize_psleep)
    df_sleep_allneurons = pd.DataFrame(sleep_data0, index = sleep_data0)    
    df_sleep_allneurons_count = pd.DataFrame(sleep_data0, index = sleep_data0)    
    
    for a, b in enumerate (idx):
	
        if b == True:
		
            aa = original_data[rat_name]['tspk'][idx][x] * 1000
            x = x + 1 
            df = pd.DataFrame(aa)
            df.columns = ['Spiking_time']
            df1 = df.apply(np.floor) 
            df1_pre_spike = df1[(df1.Spiking_time > pre_start) & (df1.Spiking_time < pre_end)]
            df1_pre_count = (df1_pre_spike - pre_start) / binsize_sleep
            df1_pre_count1 = df1_pre_count.apply(np.floor)
            
            df1_awake_spike = df1[(df1.Spiking_time > awake_start) & (df1.Spiking_time < awake_end)]
            df1_awake_count = (df1_awake_spike - awake_start) / binsize_awake
            df1_awake_count1 = df1_awake_count.apply(np.floor)

            df1_sleep_spike = df1[(df1.Spiking_time > sleep_start) & (df1.Spiking_time < sleep_end)]
            df1_sleep_count = (df1_sleep_spike - sleep_start) / binsize_psleep
            df1_sleep_count1 = df1_sleep_count.apply(np.floor)

            df1_pre_count1.insert(1,'spiking',1)
            df1_awake_count1.insert(1,'spiking',1)
            df1_sleep_count1.insert(1,'spiking',1)

            pre_data = np.arange((pre_end - pre_start) / binsize_sleep)
            df_pre = pd.DataFrame(pre_data, index = pre_data)
            df_pre = df_pre.drop([0],axis = 1)
            df_pre.insert(0,'spiking',0)
            
            df_pre_count = pd.DataFrame(pre_data, index = pre_data)
            df_pre_count = df_pre_count.drop([0],axis = 1)
            df_pre_count.insert(0,'spiking',0)
            
            df1_pre_count1[['Spiking_time']] = df1_pre_count1[['Spiking_time']].astype(int)
            df1_pre_count_sum = df1_pre_count1.groupby('Spiking_time').sum()
            df1_pre_count1 = df1_pre_count1.drop_duplicates('Spiking_time').set_index('Spiking_time')
            
            column_names = "neuron"+ str(x)

            df_pre_count.update(df1_pre_count_sum)           
            df_pre.update(df1_pre_count1)    
            
            df_pre_allneurons_count[column_names] = df_pre_count.spiking
            df_pre_allneurons[column_names] = df_pre.spiking
            
            awake_data = np.arange((awake_end-awake_start) / binsize_awake)
            df_awake = pd.DataFrame(awake_data, index = awake_data)
            df_awake = df_awake.drop([0],axis = 1)
            df_awake.insert(0,'spiking',0)
            
            df_awake_count = pd.DataFrame(awake_data, index = awake_data)
            df_awake_count = df_awake_count.drop([0],axis = 1)
            df_awake_count.insert(0,'spiking',0)
            
            df1_awake_count1[['Spiking_time']] = df1_awake_count1[['Spiking_time']].astype(int)
            df1_awake_count_sum = df1_awake_count1.groupby('Spiking_time').sum()
            df1_awake_count1 = df1_awake_count1.drop_duplicates('Spiking_time').set_index('Spiking_time')
            
            df_awake_count.update(df1_awake_count_sum)            
            df_awake.update(df1_awake_count1)
            
            df_awake_allneurons_count[column_names] = df_awake_count.spiking
            df_awake_allneurons[column_names] = df_awake.spiking
            
            sleep_data = np.arange((sleep_end-sleep_start) / binsize_psleep)
            df_sleep = pd.DataFrame(sleep_data, index = sleep_data)
            df_sleep = df_sleep.drop([0],axis = 1)
            df_sleep.insert(0,'spiking',0)
            
            df_sleep_count = pd.DataFrame(sleep_data, index = sleep_data)
            df_sleep_count = df_sleep_count.drop([0],axis = 1)
            df_sleep_count.insert(0,'spiking',0)
            
            df1_sleep_count1[['Spiking_time']] = df1_sleep_count1[['Spiking_time']].astype(int)
            df1_sleep_count_sum = df1_sleep_count1.groupby('Spiking_time').sum()
            df1_sleep_count1 = df1_sleep_count1.drop_duplicates('Spiking_time').set_index('Spiking_time')
            
            df_sleep_count.update(df1_sleep_count_sum)             
            df_sleep.update(df1_sleep_count1)
            
            df_sleep_allneurons_count[column_names] = df_sleep_count.spiking
            df_sleep_allneurons[column_names] = df_sleep.spiking
            
    df_pre_allneurons = df_pre_allneurons.drop (0,1) 
    df_pre_allneurons = df_pre_allneurons.astype(int)        
    df_pre_allneurons.index = df_pre_allneurons.index.astype(int)
    df_pre_allneurons_count = df_pre_allneurons_count.drop (0,1) 
    df_pre_allneurons_count = df_pre_allneurons_count.astype(int)        
    df_pre_allneurons_count.index = df_pre_allneurons_count.index.astype(int)
    pre_count_sum = df_pre_allneurons_count.sum(axis = 1)
    df_pre_allneurons.insert(0,'count_sum',pre_count_sum.values)
    
    df_awake_allneurons = df_awake_allneurons.drop (0,1)
    df_awake_allneurons = df_awake_allneurons.astype(int)
    df_awake_allneurons.index = df_awake_allneurons.index.astype(int)
    df_awake_allneurons_count = df_awake_allneurons_count.drop (0,1)
    df_awake_allneurons_count = df_awake_allneurons_count.astype(int)
    df_awake_allneurons_count.index = df_awake_allneurons_count.index.astype(int)	
    awake_count_sum = df_awake_allneurons_count.sum(axis = 1)
    df_awake_allneurons.insert(0,'count_sum',awake_count_sum.values)
    
    df_sleep_allneurons = df_sleep_allneurons.drop (0,1)
    df_sleep_allneurons = df_sleep_allneurons.astype(int)
    df_sleep_allneurons.index = df_sleep_allneurons.index.astype(int)  
    df_sleep_allneurons_count = df_sleep_allneurons_count.drop (0,1)
    df_sleep_allneurons_count = df_sleep_allneurons_count.astype(int)
    df_sleep_allneurons_count.index = df_sleep_allneurons_count.index.astype(int)     
    sleep_count_sum = df_sleep_allneurons_count.sum(axis = 1)
    df_sleep_allneurons.insert(0,'count_sum',sleep_count_sum.values)
    
    #clean the awake data by velocity
    
    awake_data = df_awake_allneurons
    posotion_rat = position[position_data_name]
    df_posotion_rat = pd.DataFrame(posotion_rat,columns = ['time','x','y'])
    df_posotion_rat = df_posotion_rat.drop(df_posotion_rat[(df_posotion_rat['x'] == 0) & (df_posotion_rat['y'] == 0)].index)
    
    run_time = df_posotion_rat['time'].values / 10000
    position_x = df_posotion_rat['x'].values
    position_y = df_posotion_rat['y'].values
    velocity_array = []
    velocity_i = 0
    loop = np.arange(len(df_posotion_rat))

    for a in loop:
        velocity_array.append(velocity_i)
        if a < len(df_posotion_rat) - 1:
            velocity_i = np.sqrt((position_x[a+1] - position_x[a])**2 + (position_y[a+1] - position_y[a])**2) / (run_time[a+1] - run_time[a])
        
    df_posotion_rat.insert(3,'velocity',velocity_array)
    df_posotion_rat.time = df_posotion_rat.time/1000000
    output_times = df_posotion_rat.time.values
    outputs = df_posotion_rat.velocity.values
    
    start = awake_start / 1000
    end = awake_end / 1000
    dt = binsize_awake / 1000 
    
    edges=np.arange(start,end,dt) #Get edges of time bins
    num_bins=edges.shape[0] #Number of bins
    #output_dim=outputs.shape[1] #Number of output features
    outputs_binned=np.empty([num_bins]) #Initialize matrix of binned outputs
    #Loop through bins, and get the mean outputs in those bins
    
    for i in range(num_bins): #Loop through bins
    
        if i ==0:
            outputs_binned[i] = 0
        else:
            outputs_binned[i]=np.mean(outputs[idxs])
            
        if i < num_bins - 1:
            idxs=np.where((np.squeeze(output_times)>edges[i]) & (np.squeeze(output_times)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
    
    awake_data.insert(1,'velocity',outputs_binned)
    awake_data = awake_data[awake_data.velocity >= 0.15]
    awake_data = awake_data.drop('velocity',1)
    
    return df_pre_allneurons, awake_data, df_sleep_allneurons
	
# Data exploration
def plot_NeuronSpike(data, num, **kwargs):

    n = num - 2
    data_s = data[(data.index > n) & (data.index < (data.index.values[-1]-n))]
    data_list = list(data_s.index.values)
    start = random.sample(data_list,1)[0] 
    select_index = range(start,start + num,1)
    data_num = data.loc[data.index[select_index].values]
    spikes = []
    nr = range(len(data_num.columns))
	
    for i in nr:
	
        oninds = data_num[data_num[data_num.columns[i]] == 1].index.values
        spikes.append(oninds)

    ax = plt.gca() 
	
    for ith, trial in enumerate(spikes):
	
        plt.vlines(trial, ith + .5, ith + 1, **kwargs)
    plt.ylim(0, len(spikes))
    
    return ax

# Generate the SWR replay data,simulated groud truth data

def SWR(original_data_ripple, epochs, record_name, binsize_psleep, tolerance, ripple_name, phase):
    
    if phase == 0:        
        sleep_start = epochs[record_name][0] * 1000
        sleep_end = epochs[record_name][1] * 1000
    if phase == 1:
        sleep_start = epochs[record_name][4] * 1000
        sleep_end = epochs[record_name][5] * 1000
        
    sleep_data0 = np.arange((sleep_end-sleep_start) / binsize_psleep)
    df_sleep_allneurons = pd.DataFrame(sleep_data0, index = sleep_data0)

    aa = original_data_ripple[ripple_name]* 1000
    df = pd.DataFrame(aa)
    df.columns = ['ripple_time']
    df1 = df.apply(np.floor) 

    df1_sleep_ripple = df1[(df1.ripple_time > sleep_start) & (df1.ripple_time < sleep_end)]
    xxx = []
    ripple_time_array = df1_sleep_ripple.ripple_time.values

    if tolerance >0:
        for a in range(len(df1_sleep_ripple)):
            xx = np.arange(ripple_time_array[a] - tolerance, ripple_time_array[a] + tolerance)
            xxx.append(xx)
        new_xxx = np.reshape(xxx, len(df1_sleep_ripple)*2*tolerance)
        
    if tolerance == 0:
        new_xxx = df1_sleep_ripple['ripple_time'].values
    
    df_xxx = pd.DataFrame(new_xxx, columns = ['ripple_time'])
    df1_sleep_count = (df_xxx - sleep_start) / binsize_psleep
    df1_sleep_count1 = df1_sleep_count.apply(np.floor)
    df1_sleep_count1.insert(1,'ripple', 1)
    
    sleep_data = np.arange((sleep_end-sleep_start) / binsize_psleep)
    df_sleep = pd.DataFrame(sleep_data, index = sleep_data)
    df_sleep = df_sleep.drop([0],axis = 1)
    df_sleep.insert(0,'ripple',0)
    df1_sleep_count1[['ripple_time']] = df1_sleep_count1[['ripple_time']].astype(int)
    df1_sleep_count1 = df1_sleep_count1.drop_duplicates('ripple_time').set_index('ripple_time')
    df_sleep.update(df1_sleep_count1)
    df_sleep_allneurons['pattern'] = df_sleep.ripple
    df_sleep_allneurons = df_sleep_allneurons.drop (0,1)
    df_sleep_allneurons = df_sleep_allneurons.astype(int)
    df_sleep_allneurons.index = df_sleep_allneurons.index.astype(int)
    
    return df_sleep_allneurons