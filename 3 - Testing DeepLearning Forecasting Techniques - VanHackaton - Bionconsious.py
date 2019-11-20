# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:46:11 2019

@author: arsf
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import time
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import callbacks
from ClarkeError import clarke_error_grid
import sys
import pickle
plt.rcParams['figure.figsize'] = [15, 5]

def writeCsv(df,path,sep = ';'):
    df.to_csv(path, sep = sep, index = True, mode = 'a' )

def specialPlot(regr, X_test, y_test, window_size, horizon, suffix, technique):
    #Plots the predicted and the y_test data when the horizon > 1 (by sequencing forecasts sets (window + horizon))
    i = window_size
    first = list(regr.predict(X_test[0].reshape(1, -1))[0,:])
    _y_test = list(y_test[0])
    i = i + horizon
    while i < X_test.shape[0] - horizon:
        first = first + list(regr.predict(X_test[i].reshape(1, -1))[0,:])
        _y_test = _y_test + list(y_test[i])
        i = i + horizon
        
    plt.figure()
       
    plt.plot(range(len(_y_test)), _y_test, label='Test Data')
    plt.plot(range(len(_y_test)), first, color='red', label='Prediction')   
    pylab.xlabel('Sample')
    pylab.ylabel('Y')
    pylab.title('Variables ['+suffix+'] '+ technique +' j= '+str(window_size))
    plt.legend(loc=0)
    plt.grid('on')
    
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig(os.path.join('Figures','Forecasting '+ technique +'Variables ['+suffix+'] j= '+str(window_size)+'.png'))
    plt.close()

def calculateClarke(y_test,predicted,window_size,horizon,suffix,technique):
    #Calculates and plot clarke error for predicted data > horizon
    i = window_size    
    first = list(predicted[0])
    _y_test = list(y_test[0])
    i = i + horizon
    while i < y_test.shape[0] - horizon:
        first = first + list(predicted[i])
        _y_test = _y_test + list(y_test[i])
        i = i + horizon
            
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
        
    plt, zone = clarke_error_grid(_y_test, first, 'Clarke_'+suffix+'_wSize_'+str(window_size))
    plt.savefig(os.path.join('Figures','Clarke '+ technique +'Variables ['+suffix+'] j= '+str(window_size)+'.png'))
    plt.close()
    
    labels = ['A','B','C','D','E']
    fig1, ax1 = plt.subplots()
    
    ax1.pie(zone, labels=labels,shadow=True, autopct='%1.1f%%')
    ax1.set_title('Clarke_'+suffix+'_wSize_'+str(window_size))
    plt.savefig(os.path.join('Figures','Clarke Pizza '+ technique +'Variables ['+suffix+'] j= '+str(window_size)+'.png'))
    plt.close()
    
    return ((zone[0] + zone[1])/sum(zone))*100, zone
        
def main():

    #Data Frame that will receive the RMSE Results
    dfResults = pd.DataFrame()
    
    '''For test the algorithms in the whole set of slided data, please uncomment line 91 and comment line 92.
    For that, all the data has to be generated from the script 1 - Data Exploration and Pre-...'
    In order to save your time, the current configuration only test and train models with the
    Best valuated window sizes of the techniques. '''
    
    #for window_size in range(48,150,4):
    for window_size in range(52,69,4):
        #Loop over feature selection
        for suffix in ['glucose_only','glucose_activity','glucose_activity_heart']:
            
            #Necessary to multiply the number of inputs of the model
            if suffix == 'glucose_only':
                parameters = 1
            elif suffix == 'glucose_activity':
                parameters = 2
            elif suffix == 'glucose_activity_heart':
                parameters = 3
                
            #Build the file names
            fileX = 'slided_window_size_'+str(window_size)+'_X_'+suffix+'.pkl'
            fileY = 'slided_window_size_'+str(window_size)+'_Y_'+suffix+'.pkl'
            
            #Loads the data
            if os.path.exists(os.path.join('Data',fileX)):
                with open(os.path.join('Data',fileX), 'rb') as handle:
                    pipeX = pickle.load(handle) 
                with open(os.path.join('Data',fileY), 'rb') as handle:
                    pipeY = pickle.load(handle)
            else:
                print('Error: .pkl file not Found')
                sys.exit()
            
            #Takes Two Weeks as training
            X_train = pipeX[0:4032].fillna(0).values
            y_train = pipeY[0:4032].fillna(0).values
                        
            #Takes 5 weeks as testing
            X_test = pipeX[2030:12109].fillna(0).values
            y_test = pipeY[2030:12109].fillna(0).values
            
            #LSTM
            technique = 'LSTM'
            print('Fitting: '+technique+'... window: '+str(window_size))
            
            #Scaling Data
            shape = X_train.shape
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train.reshape(-1,1)).reshape(shape)
            shape = y_train.shape
            scaled_y_train = scaler.transform(y_train.reshape(-1,1)).reshape(shape)
            shape = X_test.shape
            scaled_x_test = scaler.transform(X_test.reshape(-1,1)).reshape(shape)
            shape = y_test.shape
            #scaled_y_test = scaler.transform(y_test.reshape(-1,1)).reshape(shape)
            
            X_train = scaled_X_train.reshape((-1,1,window_size*parameters))
            X_test = scaled_x_test.reshape((-1,1,window_size*parameters))
            
            try:
                model = Sequential()
                model.add(LSTM(56, input_shape=(1, window_size*parameters)))
                model.add(Dropout(rate=0.2))
                model.add(Dense(units=12))                
                model.compile(loss='mse', optimizer='adam', metrics=['mse'])
                
                if not os.path.exists('tensorboard'):
                    os.makedirs('tensorboard')
                
                tensorboard = TensorBoard(log_dir="tensorboard/{}".format(time.time()))
                
                early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=12)
                
                # tensorboard --logdir="" --host localhost --port 8088
                
                model.fit(X_train,scaled_y_train,batch_size=50, epochs=100, validation_split=0.1, verbose=1, callbacks=[tensorboard,early_stop])

                predicted = model.predict(X_test)

                #Brings back the original distribution
                shape = predicted.shape
                predicted = scaler.inverse_transform(predicted.reshape(-1,1)).reshape(shape)
                clarke, zone = calculateClarke(y_test,predicted,window_size,12,'Clarke_'+suffix+'_'+technique+'_wSize_'+str(window_size),technique)
                error = sqrt(mean_squared_error(predicted, y_test))
                specialPlot(model, predicted, y_test, window_size, 12, suffix, technique)
            except:
                error = float('nan')
            
            dfResults = dfResults.append({'technique':technique,
                                          'parameters':'',
                                          'variables':suffix,
                                          'window_size':window_size,
                                          'clarke':clarke,
                                          'zone':zone,
                                          'RMSE':error},ignore_index = True)
    
    #Writes the results as .csv
    writeCsv(dfResults,'results_deep_learning.csv')

if __name__ == '__main__':
    main()