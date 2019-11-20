# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:46:11 2019

@author: arsf
"""
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import os
import numpy as np
import pickle
plt.rcParams['figure.figsize'] = [15, 5]

def Gap_check(index, maxGap):
    #Checks if there is any gap between the pairs of timestamps that in bigger than 'maxGap' in minutes
    for i in range(0,len(index)-1):
        if (pd.Timedelta(index[i+1] - index[i]).seconds)/60 > maxGap:
            return True
    return False

def PipelineWindow(mainSerie, dfs, window_size,horizon,suffix):
    #Sliding window process
    fileX = 'slided_window_size_'+str(window_size)+'_X_'+suffix+'.pkl'
    fileY = 'slided_window_size_'+str(window_size)+'_Y_'+suffix+'.pkl'
    
    if not os.path.exists('Data'):
        os.makedirs('Data')
    
    if os.path.exists(os.path.join('Data',fileX)):
        with open(os.path.join('Data',fileX), 'rb') as handle:
            pipeX = pickle.load(handle) 
        with open(os.path.join('Data',fileY), 'rb') as handle:
            pipeY = pickle.load(handle)         
        
    else:
        
        pipeX = pd.DataFrame()
        pipeY = pd.DataFrame()
        
        for i in tqdm(range(0,len(mainSerie)+1,1)): #Until the end of the series minus windows_size + 1 
            if len(mainSerie) - i > window_size+horizon and Gap_check(mainSerie[i:i+window_size].index,7) == False: #fits another instance
                #If there is more than one variable, put them together in the instance                
                dfTempX = mainSerie[i:i+window_size].values
                            
                for k in dfs:
                    dfTempX = np.append(dfTempX, k[i:i+window_size])
                                                            
                dfTempY = pd.Series(mainSerie[i+window_size:(i+window_size)+(horizon)].values)
                pipeX = pipeX.append(pd.Series(dfTempX),ignore_index = True)
                pipeY = pipeY.append(dfTempY,ignore_index = True)
    
        with open(os.path.join('Data',fileX), 'wb') as handle:
            pickle.dump(pipeX, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        with open(os.path.join('Data',fileY), 'wb') as handle:
            pickle.dump(pipeY, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
    return pipeX,pipeY

def aggregate(target, to_agg, col, agg):
    #Aggregates data based on either sum or average
    last = target[0]
    new = [0]
    print('Aggregating data...')
    for i in tqdm(range(1,len(target))):
        if agg == 'sum':
            new.append(sum(to_agg.loc[(to_agg.index > last) & (to_agg.index <= target[i]),col]))
        elif agg == 'mean':
            new.append(np.mean(to_agg.loc[(to_agg.index > last) & (to_agg.index <= target[i]),col]))
        last = target[i]
    return new

def main():

    #Load Data
    dfGlucose = pd.read_csv(os.path.join('blood-glucose-data.csv'), sep=',',index_col = 'point_timestamp' ,error_bad_lines=True)
    dfActivity = pd.read_csv(os.path.join('distance-activity-data.csv'), sep=',',index_col = 'point_timestamp' ,error_bad_lines=True)
    dfHeart = pd.read_csv(os.path.join('heart-rate-data.csv'), sep=',',index_col = 'point_timestamp' ,error_bad_lines=True)
    
    #Convert point_value to numeric
    dfGlucose['point_value(mg/dL)'] = dfGlucose['point_value(mg/dL)'].astype(dtype=float)
    dfActivity['point_value(kilometers)'] = dfActivity['point_value(kilometers)'].astype(dtype=float)
    dfHeart['point_value'] = dfHeart['point_value'].astype(dtype=float)
    
    #Convert index to time/data
    dfGlucose.index = pd.to_datetime(dfGlucose.index) 
    dfActivity.index = pd.to_datetime(dfActivity.index)
    dfHeart.index = pd.to_datetime(dfHeart.index)
    
    #Aggregate data of Activity and Heart
    #Activity
    Activity = aggregate(dfGlucose.index, dfActivity, 'point_value(kilometers)', 'sum')
    #Heart
    Heart = aggregate(dfGlucose.index, dfHeart, 'point_value', 'mean')
    
    #Plot and save the glucose values to explore Data
    plt.figure()
    plt.plot(dfGlucose.index, dfGlucose['point_value(mg/dL)'].values, label='Glucose')    
    pylab.xlabel('Timestamp')
    pylab.ylabel('Glucose (mg/dL)')
    plt.legend(loc=0)
    plt.grid('on')
    plt.savefig("Glucose_point_value(mg-dL).png")
    plt.close()    
    
    #Plot and save the Activity values to explore Data    
    plt.figure()
    plt.plot(dfGlucose.index, Activity, label='Aggregated Activity')    
    pylab.xlabel('Timestamp')
    pylab.ylabel('Agg Activity')
    plt.legend(loc=0)
    plt.grid('on')
    plt.savefig("Activity_aggregated.png")
    plt.close() 
    
    #Plot and save the Heart values to explore Data
    plt.figure()
    plt.plot(dfGlucose.index, Heart, label='Aggregated Heart')    
    pylab.xlabel('Timestamp')
    pylab.ylabel('Agg Heart')
    plt.legend(loc=0)
    plt.grid('on')
    plt.savefig("Heart_aggregated.png")
    plt.close() 
        
    #Separating instances of forecasting (discarting weeks with missing data)
    
    '''For generate the whole set of slided data uncomment line 130 and comment line 131.
    In order to save your time, the current configuration only generates data with the
    Best valuated window sizes of the techniques. '''
    
    #for window_size in range(48,150,4):
    for window_size in range(52,69,4):
        horizon = 12
        #Transforming the data in instances of regression (univariate)
        print('Sliding window_size: '+str(window_size)+' for glucose_only')
        PipelineWindow(dfGlucose['point_value(mg/dL)'],[],window_size,horizon,'glucose_only')
        print('Sliding window_size: '+str(window_size)+' for glucose_activity')
        PipelineWindow(dfGlucose['point_value(mg/dL)'],[Activity],window_size,horizon,'glucose_activity')
        print('Sliding window_size: '+str(window_size)+' for glucose_activity_heart')
        PipelineWindow(dfGlucose['point_value(mg/dL)'],[Activity,Heart],window_size,horizon,'glucose_activity_heart')
    
        #The Pipeline Window is in charge of saving them to file if they are not there yet.
    
if __name__ == '__main__':
    main()