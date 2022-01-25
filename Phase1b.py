
#%%

## General libraries
## General libraries
import numpy as np # Numpy
import pandas as pd # Pandas
import pandas.plotting
import pickle as pk # Pickles
import datetime as datetime
import os
import logging
import sys
import astral
import scipy
import pytz
import tsfresh

#import heatmapz
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from astral import LocationInfo
from astral.sun import sun

# %% Generate Phase1b dataset

dw_directory = "./data"

no_stations = np.linspace(201, 275, 75)

dataset = pd.DataFrame()


for i in no_stations:
    # # Read dataset
    filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')
    #if os.path.exists(filepath):
        # Read .txt file
    with open(filepath, 'r') as f:
         data_v = pd.read_csv(f)
         
    if len(dataset) == 0:
        dataset = data_v
    else:
        dataset = dataset.append(data_v)

        
filepath = os.path.join(dw_directory, 'test.csv')
#if os.path.exists(filepath):
# Read .txt file
with open(filepath, 'r') as f:
    final_test = pd.read_csv(f)       


#%% clean

dataseta = dataset.drop(['weekday'], axis=1)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(dataseta)   

dataseta = pd.DataFrame(imp_mean.transform(dataseta), columns=dataseta.columns)

dataseta['weekday'] = dataset['weekday'].reindex().tolist()

dataseta = dataseta.dropna()

# %% Find correlations        
 
#features = pd.DataFrame(data_v.columns)
     
# calculate pearsons coefficient for features
corr = data_v.corr(method = 'pearson')

# plot heatmap
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

enc = OneHotEncoder(handle_unknown='ignore')
cols_d = np.array(dataseta['weekday']).reshape(-1,1)
enc.fit(cols_d)
cols = enc.categories_[0].tolist()
cols_d = pd.DataFrame(enc.transform(cols_d).toarray(), columns=cols)

datasetb = dataseta.drop(['weekday'], axis=1)

datasetb =  pd.concat([datasetb, cols_d], axis=1).reindex(datasetb.index)
#%%
pr = []
for i in datasetb.columns:
    x = scipy.stats.pearsonr(datasetb[i], dataseta['bikes'])[0]
    # print(type(pr))
    if len(pr) == 0:
        pr = [x]
    else:    
    # #pr = [pr,x]
        pr.append(x)

#%%
fig = plt.barh(datasetb.columns, pr)
#fig.tight_layout()
#%%
#plt.hist(datasetb['hour'], datasetb['bikes'])
n, bins, patches = plt.hist(datasetb['bikes'], 50, density=True, facecolor='g', alpha=0.75)

#%% 
#sns.pairplot(datasetb)

#%% feature engineering

## is it dark?

city = LocationInfo(39.4502730411, -0.3333629598)

dk = pd.to_datetime(dataseta['timestamp'], unit='s')
darkness = []
for i in dk:
    s = sun(city.observer, date=i)
    srise = s['sunrise'].replace(tzinfo=None)
    sset = s['sunset'].replace(tzinfo=None)
    if i < sset and i > srise:
        d = 0
    else:
        d = 1
    if len(darkness) == 0:
        darkness = [d]
    else:
        darkness.append(d)

datasetb['darkness'] = darkness

#%%
## distance to another station
import geopy.distance
from scipy import spatial

stations_lat = pd.unique(datasetb['latitude'])
stations_long = pd.unique(datasetb['longitude'])

stations = np.stack((stations_lat, stations_long), axis=1)

no_s = np.linspace(0,74, 75)

near = []
for si1 in no_s:
    distance = []
    for si2 in no_s:
        if si1 == si2:
            continue
        dist = geopy.distance.geodesic(stations[int(si1),:], stations[int(si2),:]).km
        if len(distance) ==0:
            distance = [dist]
        else:
            distance.append(dist)
    nearest = min(distance)
    if len(near) == 0:
        near = [nearest]
    else:
        near.append(nearest)


#%%
def label_dist (row):
    x = near[int(row['station'])-int(201)]
    return x

datasetb['distance'] = datasetb.apply (lambda row: label_dist(row), axis=1)
## start and end of work/school

#%% timeseries
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
settings = ComprehensiveFCParameters()

def label_dist2 (row):
    x = extracted_features
    x = x.iloc[int(row['day'])-1]
    return x

#no_stations1 = np.linspace(201,204,4)

data_ts = pd.DataFrame()

for sta in no_stations:
    data_t = datasetb[datasetb['station']==sta]
    data_time = data_t[['day', 'hour', 'bikes_3h_ago']]
    print(sta)
    extracted_features = extract_features(data_time, column_id="day", column_sort="hour", default_fc_parameters=settings, n_jobs=0)

    #for i in extracted_features.columns:
    #    datasetb[i] = datasetb.apply (lambda row: label_dist2(row, i), axis=1)
    cols = extracted_features.columns.tolist()
    data_t[cols] = data_t.apply (lambda row: label_dist2(row), axis=1)

    if len(data_ts) == 0:
        data_ts = data_t
    else:
        data_ts = data_ts.append(data_t)


#%% Feature Importance

data_ts = data_ts.dropna(axis=1)

#%%
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
#dataset_X, dataset_y = load_breast_cancer(return_X_y=True)

dataset_X = data_ts.drop(['bikes'], axis=1).copy()
dataset_y = data_ts['bikes'].astype(float)

scaler = MinMaxScaler()
scaler.fit(dataset_X)
dataset_X = pd.DataFrame(scaler.transform(dataset_X), columns=dataset_X.columns)

# X_new = mutual_info_regression(dataset_X, dataset_y)

pr = []
for i in dataset_X.columns:
    x = scipy.stats.pearsonr(dataset_X[i], dataset_y)[0]
    # print(type(pr))
    if len(pr) == 0:
        pr = [x]
    else:    
    # #pr = [pr,x]
        pr.append(x)
#%%
prn = pd.DataFrame(pr, columns=['value'])
prn['feature'] = dataset_X.columns

#%%
impfeat1 = prn[prn['value'] > 0.5]
impfeat2 = prn[prn['value'] < -0.5]
impfeat= impfeat1.append(impfeat2)
impfeat = impfeat['feature'].tolist()

# %% Pipeline
dataset_X = datasetb.drop(['bikes'], axis=1).copy()
dataset_y = datasetb['bikes'].copy()

#%%

d_X = dataset_X[impfeat]
#%%
# Break off validation set from training data
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, 
        
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]

#Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
#X_valid = X_valid_full[my_cols].copy()
X_test = X_test[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#X_train.isna().any()
#%%
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
# Define model


model1 = RandomForestRegressor(n_estimators=100, random_state=0)
model2 = SGDRegressor(max_iter=1000000, tol=1e-3, learning_rate='optimal')
model3 = linear_model.BayesianRidge()
model4 = AdaBoostRegressor(random_state=0, n_estimators=500)
model5 = ExtraTreesRegressor(n_estimators=100, random_state=0)
model6 = BaggingRegressor(base_estimator=SVR(),
                                 n_estimators=10, random_state=0)
models = [model1]
#models = [model2]
# Bundle preprocessing and modeling code in a pipeline
n = 0
for i in models:
    n = n+1
    clf = Pipeline(steps=[('model', i)
                     ])

# Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
    preds = clf.predict(X_test)

    print('MAE using model {}'.format(str(n)), mean_absolute_error(y_test, preds))
# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)

# %% Final Test


enc = OneHotEncoder(handle_unknown='ignore')
cols_d = np.array(final_test['weekday']).reshape(-1,1)
enc.fit(cols_d)
cols = enc.categories_[0].tolist()
cols_d = pd.DataFrame(enc.transform(cols_d).toarray(), columns=cols)

f_d = final_test.drop(['weekday'], axis=1)

f_d =  pd.concat([f_d, cols_d], axis=1).reindex(f_d.index)

#%% feature engineering

## is it dark?

city = LocationInfo(39.4502730411, -0.3333629598)

dk_f = pd.to_datetime(f_d['timestamp'], unit='s')
darkness = []
for i in dk_f:
    s = sun(city.observer, date=i)
    srise = s['sunrise'].replace(tzinfo=None)
    sset = s['sunset'].replace(tzinfo=None)
    if i < sset and i > srise:
        d = 0
    else:
        d = 1
    if len(darkness) == 0:
        darkness = [d]
    else:
        darkness.append(d)

f_d['darkness'] = darkness

#%%
## distance to another station
import geopy.distance
from scipy import spatial

stations_lat = pd.unique(f_d['latitude'])
stations_long = pd.unique(f_d['longitude'])

stations = np.stack((stations_lat, stations_long), axis=1)

no_s = np.linspace(0,74, 75)

near = []
for si1 in no_s:
    distance = []
    for si2 in no_s:
        if si1 == si2:
            continue
        dist = geopy.distance.geodesic(stations[int(si1),:], stations[int(si2),:]).km
        if len(distance) ==0:
            distance = [dist]
        else:
            distance.append(dist)
    nearest = min(distance)
    if len(near) == 0:
        near = [nearest]
    else:
        near.append(nearest)


#%%
def label_dist (row):
    x = near[int(row['station'])-int(201)]
    return x

f_d['distance'] = f_d.apply (lambda row: label_dist(row), axis=1)


#%% timeseries
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
settings = ComprehensiveFCParameters()

def label_dist2 (row):
    x = extracted_features
    #print(row)
    #x = x.iloc['index'][int(row['day'])]
    x = x.iloc[int(row['day']),:]
    #print(x.index)
    return x

#no_stations1 = np.linspace(201,204,4)

data_ts = pd.DataFrame()

for sta in no_stations:
    data_t = f_d[f_d['station']==sta]
    data_time = data_t[['day', 'hour', 'bikes_3h_ago']]
    print(sta)
    extracted_features = extract_features(data_time, column_id="day", column_sort="hour", default_fc_parameters=settings, n_jobs=0)
    #extracted_features = extracted_features.reindex()

    #for i in extracted_features.columns:
    #    datasetb[i] = datasetb.apply (lambda row: label_dist2(row, i), axis=1)
    cols = extracted_features.columns.tolist()
    data_t[cols] = data_t.apply (lambda row: label_dist2(row), axis=1)

    if len(data_ts) == 0:
        data_ts = data_t
    else:
        data_ts = data_ts.append(data_t)
#%%
x = extracted_features.iloc[int(5),:]

def label_dist2 (row):
    x = extracted_features
    x = x.iloc[int(row['day'])-1]
    return x

#no_stations1 = np.linspace(201,204,4)

data_ts = pd.DataFrame()

for sta in no_stations:
    data_t = datasetb[datasetb['station']==sta]
    data_time = data_t[['day', 'hour', 'bikes_3h_ago']]
    print(sta)
    extracted_features = extract_features(data_time, column_id="day", column_sort="hour", default_fc_parameters=settings, n_jobs=0)

    #for i in extracted_features.columns:
    #    datasetb[i] = datasetb.apply (lambda row: label_dist2(row, i), axis=1)
    cols = extracted_features.columns.tolist()
    data_t[cols] = data_t.apply (lambda row: label_dist2(row), axis=1)

    if len(data_ts) == 0:
        data_ts = data_t
    else:
        data_ts = data_ts.append(data_t)

#%%
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
#dataset_X, dataset_y = load_breast_cancer(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(f_d)
f_d = pd.DataFrame(scaler.transform(f_d), columns=f_d.columns)



d_X = f_d[impfeat]
# %%

preds = pd.DataFrame(clf.predict(d_X))
# %%
preds.index+=1
# %%

preds.to_csv('submission.csv', header=['bikes'])
# %%
