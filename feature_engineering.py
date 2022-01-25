# %%

from multiprocessing.sharedctypes import Value
import numpy as np
import os
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

# User-defined
from utilities import data_saver
from fe_utilities import interpolation, weekday_handler, darkness, pca_app, station_proximity

data_group = 'A_individual'
#GROUP D
#  features = (['station','latitude','longitude','darkness',
#             'numDocks','bikes_3h_ago','weekhour','weekend', 
#             'full_profile_3h_diff_bikes','full_profile_bikes'])
short= ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes']
short_temp = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'temperature.C']
full = ['bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes']
full_temp = ['bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'temperature.C']
short_full = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'full_profile_3h_diff_bikes', 'full_profile_bikes']
short_full_temp = ['bikes_3h_ago', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'temperature.C']


features = (['station', 'latitude', 'longitude', 'numDocks', 
'timestamp','year', 'month', 'day', 'hour', 'Monday', 'Tuesday', 
'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
'weekhour', 'isHoliday', 'windMaxSpeed.m.s', 'windMeanSpeed.m.s', 
'windDirection.grades', 'temperature.C', 'relHumidity.HR', 
'airPressure.mb', 'precipitation.l.m2', 'bikes_3h_ago', 
'full_profile_3h_diff_bikes','full_profile_bikes', 
'short_profile_3h_diff_bikes','short_profile_bikes'])

    # if Test_flag == True:
load_config = {"Group"               :'A_individual',
               "Features"            :features,
               "Test"                :True,
               "Interpolation Method":'sImpute', # "sImpute" or "delete"
               "Weekday Method"      :'dotw',    # 'dotw' or 'wk_wknd'
               "Light_Dark"          :False,
               "Station Proximity"   :False,
               "Scale Data"          :False}

# # GROUP E
# features = (['airPressure.mb', 'bikes_3h_ago', 'day', 'full', 
# 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'full_temp', 
# 'hour', 'isHoliday', 'latitude', 'longitude', 'month', 'numDocks', 
# 'precipitation.l.m2', 'relHumidity.HR', 'short', 'short_full', 
# 'short_full_temp', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 
# 'short_temp', 'station', 'temperature.C', 'timestamp', 'weekend', 'weekhour', 
# 'windDirection.grades', 'windMaxSpeed.m.s', 'windMeanSpeed.m.s', 'year'])
# Save or no
saveMode = True
testFlag = True

# Configure dataset generation
interpolationMethod = 'sImpute' # "sImpute" or "delete"
weekdayMethod = 'dotw' # 'dotw' or 'wk_wknd'
daylight_switch = False
stationProximity_switch = False
scale_switch = False

# load_config = {"Group"               :'F_individual',
#                "Features"            :features,
#                "Test"                :testFlag,
#                "Interpolation Method":'sImpute', # "sImpute" or "delete"
#                "Weekday Method"      :weekdayMethod,    # 'dotw' or 'wk_wknd'
#                "Light_Dark"          :False,
#                "Station Proximity"   :False,
#                "Scale Data"          :scale_switch}



#Perform correlation studies
pca_switch = False  # perform PCA analysis & display results
pearson_switch = False # perform pearson correlation 

# Set load/save path 
dw_directory = "./data" # Set data directory

# Generating test set
if testFlag == True:
    filepath = os.path.join(dw_directory, 'test.csv')
    #if os.path.exists(filepath):
    # Read .txt file
    with open(filepath, 'r') as f:
        dataset = pd.read_csv(f)       
else:
    # Pull all stations into single df for pre-processing
    stations = np.linspace(201, 275, 75)
    dataset = pd.DataFrame()

    for i in stations:
        filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')

        with open(filepath, 'r') as f:
            data_v = pd.read_csv(f)
            
            if len(dataset) == 0:
                dataset = data_v
            else:
                dataset = dataset.append(data_v)



filepath = "/Users/jonathanerskine/Courses/Machine Learning Paradigms/CWK/MB_2021/data/USER/Models/OptimisedLinear/"
fname = 'opt_linear'

with open(os.path.join(filepath, fname), 'rb') as f:
    lin_mod = pickle.load(f)

#write script which appends estimated bike numbers from linear model (full)

model_types = ['full_temp','full','short_full_temp','short_full','short_temp','short']
test_model = ['full_temp']

model_array = []
y_s = []
for m in model_types:
    exec('model_array.append(' + m + ')')

# %%
for i in range(0,len(model_types)):

    x = dataset[model_array[i]].copy()
    mod_str = 'rlm_' + model_types[i]
    y_pred = [lin_mod[mod_str]['Intercept'].values[0]]*len(x)

    for j in range(0,len(model_array[i])):
        x_comp = [x_c * lin_mod[mod_str][model_array[i][j]].values[0] for x_c in x[model_array[i][j]].values.tolist()]
        y_pred = [sum(y_c) for y_c in zip(y_pred,x_comp)]
    
    # dataset[model_types[i]] = y_pred


# %% Encode 'weekday' as one-hot weekday encodings
enc = OneHotEncoder(handle_unknown='ignore')
days = np.array(dataset['weekday']).reshape(-1,1)
enc.fit(days)
cols = enc.categories_[0].tolist()
days = pd.DataFrame(enc.transform(days).toarray(), columns=cols)

less_significant_columns = ['relHumidity.HR','windDirection.grades','hour','day']
# dataset = dataset.drop(less_significant_columns,axis=1)

dataset = dataset.drop(['weekday'], axis=1)
if testFlag == False:
    dataset_y = pd.DataFrame(dataset['bikes'].copy())
    dataset = dataset.drop(['bikes'],axis=1)
else:
    dataset = dataset.drop(["Id"],axis=1)

# %% Impute or delete nan rows

dataset = interpolation(dataset, interpolationMethod)
if testFlag == False:
    dataset_y = interpolation(dataset_y, interpolationMethod)
# %% Handling days of the week - one hot encoding per day or week/weekend
dataset = weekday_handler(dataset,weekdayMethod,days)

# %% Calculates sunrise and sunset to give -> is it dark?
if daylight_switch == True:
    dataset['darkness'] = darkness(dataset)

#%% Distance to another station
if stationProximity_switch == True:
    dataset['distance'] = station_proximity(dataset)

if scale_switch == True:
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)

# redundant_columns = (['year','month','precipitation.l.m2',
#                         'short_profile_3h_diff_bikes','short_profile_bikes',
#                         'timestamp','day', 'hour',
#                         'isHoliday','windMaxSpeed.m.s','windMeanSpeed.m.s',
#                         'windDirection.grades','temperature.C','relHumidity.HR',
#                         'airPressure.mb'])
# dataset = dataset.drop(redundant_columns,axis =1)

#%%
if sorted(features) == sorted(dataset.columns.tolist()):
    pass
else:
    print(sorted(features))
    print(sorted(dataset.columns.tolist()))
    raise ValueError("Features do not match those specified")
    
# %%
config = {"Group"               :data_group,
          "Features"            :features,
          "Test"                :testFlag,
          "Interpolation Method":interpolationMethod,
          "Weekday Method"      :weekdayMethod,
          "Light_Dark"          :daylight_switch,
          "Station Proximity"   :stationProximity_switch,
          "Scale Data"          :scale_switch}


if saveMode == True:
    if testFlag == False:
        dataSETS = [dataset,dataset_y]
        xy_dict = {"Name":['X','Y'],"Data":dataSETS}
    elif testFlag == True:
        dataSETS = [dataset]
        xy_dict = {"Name":['X_test'],"Data":dataSETS}
    else:
        raise ValueError("testFlag should be True or False")
    
    for i in range(0,len(dataSETS)):
        df = xy_dict["Data"][i]
        XorY = xy_dict["Name"][i]

        all_stations = df
        save_list = [config,all_stations]

        if 'X' in XorY:
            stations = pd.unique(df['station']) # Return ID of indiviual stations

            for station in stations:
                idx = dataset.index[df['station'] == station].tolist()
                # df = df.drop add a line here for dropping things like station and location
                save_list.append(df.iloc[idx])

            data_saver(config,save_list,XorY)
        elif XorY == 'Y':
            stations = pd.unique(xy_dict["Data"][i-1]['station']) # Return ID of indiviual stations

            for station in stations:
                idx = dataset.index[xy_dict["Data"][i-1]['station'] == station].tolist()
                save_list.append(df.iloc[idx])

            
            data_saver(config,save_list,XorY)
        else:
            raise ValueError("XorY must be 'X' or 'Y'")
 # %%  

# %% Correlation analysis 
if pearson_switch == True:
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

    fig = ax.get_figure()  

    pr = []
    for i in dataset.columns:
        x = scipy.stats.pearsonr(dataset[i], dataset_y['bikes'])[0]
        
        if len(pr) == 0:
            pr = [x]
        else:    
        # #pr = [pr,x]
            pr.append(x)
    n, bins, patches = plt.hist(dataset_y['bikes'], 50, density=True, facecolor='g', alpha=0.75)
    fig = plt.barh(datasetb.columns, pr)
    # sns.pairplot(datasetb)

#%%
if pca_switch == True:
    pca_app(dataset,dataset_y,2,10)
