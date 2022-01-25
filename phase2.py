#%%
# %load_ext autoreload
# %autoreload 2

## General libraries
from distutils.archive_util import make_tarball
import numpy as np # Numpy
import pandas as pd # Pandas
import pandas.plotting
import pickle # Pickles
import os
import logging
import sys
import scipy
import pytz
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

## Usmodel_train_eval
from model_train_eval import bike_inference, bike_trainer
from utilities import data_loader, data_saver

Train_flag = True
Test_flag = True
# Group E
features = (['station', 'latitude', 'longitude', 'numDocks', 
'timestamp','year', 'month', 'day', 'hour', 'Monday', 'Tuesday', 
'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
'weekhour', 'isHoliday', 'windMaxSpeed.m.s', 'windMeanSpeed.m.s', 
'windDirection.grades', 'temperature.C', 'relHumidity.HR', 
'airPressure.mb', 'precipitation.l.m2', 'bikes_3h_ago', 
'full_profile_3h_diff_bikes','full_profile_bikes', 
'short_profile_3h_diff_bikes','short_profile_bikes'])

# stations = np.linspace(201, 275, 75)
# dataset = pd.DataFrame()

# for i in stations:
#     filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')

#     with open(filepath, 'r') as f:
#         data_v = pd.read_csv(f)
        
#         if len(dataset) == 0:
#             dataset = data_v
#         else:
#             dataset = dataset.append(data_v)



# Group D
# features = (['station','latitude','longitude','darkness',
#             'numDocks','bikes_3h_ago','weekhour','weekend', 
#             'full_profile_3h_diff_bikes','full_profile_bikes'])
# %% Load Dataset
load_config = {"Group"               :'E_individual',
               "Features"            :features,
               "Test"                :False,
               "Interpolation Method":'sImpute', # "sImpute" or "delete"
               "Weekday Method"      :'dotw',    # 'dotw' or 'wk_wknd'
               "Light_Dark"          :False,
               "Station Proximity"   :False,
               "Scale Data"          :False}

X, individual_stations_X = data_loader(load_config,'X')
Y, individual_stations_Y = data_loader(load_config,'Y')


short= ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes']
short_temp = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'temperature.C']
full = ['bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes']
full_temp = ['bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'temperature.C']
short_full = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'full_profile_3h_diff_bikes', 'full_profile_bikes']
short_full_temp = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'temperature.C']
# %% MODELS

# kernel1 = 1.0 * Matern(length_scale=1.0,length_scale_bounds=(1e-5,100000), nu=0.5)
# kernel2 = WhiteKernel(noise_level=2.0)
# kernel = kernel1 + kernel2
# model = GaussianProcessRegressor(kernel=kernel,random_state=0)

# model = DecisionTreeRegressor(min_samples_leaf=10, random_state=0)
# model = RandomForestRegressor(n_estimators=100,max_features = 3,min_samples_leaf=10, random_state=0)
#%% LINEAR MODELS
model_types = ['full_temp','full','short_full_temp','short_full','short_temp','short']

model_array = []
for m in model_types:
    exec('model_array.append(' + m + ')')
# %%
models = []
n_models = 200

for i in range(0,n_models):
    m_array = []
    for m in model_types:
        filepath = os.path.join('data/Models/Models/',
        'model_station_' + str(int(i+1)) + '_rlm_' + m + '.csv')
        #if os.path.exists(filepath):
            # Read .txt file    
        with open(filepath, 'r') as f:
            m_array.append(pd.read_csv(f))
    
    models.append(pd.DataFrame({'idx':model_types, 'models':m_array}))
        # models[m][i] = model_d



# %% PHASE 1A: Individually Trained Models

t_out = {"predictions":{el:[] for el in model_types},"MAE":{el:[] for el in model_types}}
v_out = {"predictions":{el:[] for el in model_types},"MAE":{el:[] for el in model_types}}

test_x,val_x,test_y,val_y = train_test_split(X,Y, 
                                    train_size=0.8, 
                                    test_size=0.2,
                                    random_state=0)
# %%
# Average approach
'''
Input: Models, model types
Output: 6 models, each an average of the 200 stations per model
'''
resList = []
for i in range(0,len(model_types)):
    # Get feature type length & names
    #create 200*f_length array
    # Average each column and output new dict
    m = model_types[i]
    f_n = len(model_array[i]) + 1 # + 1 for intercept
    print(model_array[i])
    resArray = [0] * f_n
    aveFactor = [1/n_models]*f_n
    print(models[0]['idx'][i])

    for j in range(0,n_models):
        resArray = [sum(x) for x in zip(resArray,
                                    [models[j]['models'][i]['weight'][n] 
                                    for n in range(0,f_n)])]
        print(resArray)
    resArray = [a * b for a, b in zip(resArray, aveFactor)]
    featureList = ['intercept'] + model_array[i]
    resDict = {'features':featureList,'weights':resArray}
    resList.append(resDict)

# %%

for i in range(0,len(model_types)):

    y = Y['bikes'].values
    x = X[model_array[i]].copy()

    y_pred = [resList[i]['weights'][0]]*len(x)

    for j in range(0,len(model_array[i])):
        x_comp = [x_c * resList[i]['weights'][j] for x_c in x[model_array[i][j]].values.tolist()]
        y_pred = [sum(y_c) for y_c in zip(y_pred,x_comp)]

    print(max(y),max(y_pred))
    err = y_pred - y
    print(max(err))
    abs_error = [abs(ele) for ele in np.subtract(y_pred,y)]
    x_p = np.linspace(0,1,len(abs_error))
    mAE = sum(abs_error)/len(x_p)
    print(mAE)
    plt.plot(x_p,abs_error,label = m,marker='.',alpha = 0.1)
    plt.show()


# %%
