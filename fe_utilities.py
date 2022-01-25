from astral import LocationInfo
from astral.sun import sun
import dash
from dash import dcc
from dash import html
import geopy.distance
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import spatial
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
 
def interpolation(dataset,interpolationMethod):
    
    if type(dataset) == type(pd.DataFrame):
        dataset = dataset.values



    if interpolationMethod == "sImpute":
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')      
        imp_mean.fit(dataset)
        output = pd.DataFrame(imp_mean.transform(dataset), columns=dataset.columns)

    elif interpolationMethod == 'delete':
        pre_del = len(dataset)
        output = dataset.dropna #(how='any', inplace=True)
        post_del = len(dataset)
        print('Warning: ',(pre_del - post_del)," entries removed from dataset.")
    else: 
        print("Error: Interpolation Method not recognised")
        exit

    return output

def weekday_handler(dataset,weekdayMethod,days):
    if weekdayMethod == 'dotw':
        output = pd.concat([dataset, days.reindex(dataset.index)], axis=1)

    elif weekdayMethod == 'wk_wknd':
        wk = ['Monday','Tuesday','Wednesday','Thursday','Friday']
        wknd = ['Saturday','Sunday']
        
        # wk_array = [0]*len(days)
        # for day in wk:
        #     wk_array = [x + y for x,y in zip(wk_array,days[day])]
        
        wknd_array = [0]*len(days)

        for day in wknd:
            wknd_array = [x + y for x,y in zip(wknd_array,days[day])]

        # wk_wknd_df = pd.DataFrame({'week': wk_array, 'weekend': wknd_array})
        wk_wknd_df = pd.DataFrame({'weekend': wknd_array})
        output = pd.concat([dataset, wk_wknd_df.reindex(dataset.index)], axis=1)
        #cols_wk = pd.DataFrame()

        wk_wknd_plot = False

        if wk_wknd_plot == True:
            x = []
            weekDays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            for day in weekDays:
                idx = [i for i, e in enumerate(days[day]) if e != 0]
                
                x.append(np.mean([dataset_y['bikes'].values[i]for i in idx]))

            ticks = list(range(0, 7)) 
                # labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
            plt.xticks(ticks, weekDays)
            plt.plot(ticks,x)
    else: 
        print("Error: Method for handling days of the week not recognised")
        exit
    
    return output

def darkness(dataset):
    city = LocationInfo(39.4502730411, -0.3333629598)

    dk = pd.to_datetime(dataset['timestamp'], unit='s')
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

    return darkness

def pca_app(dataset,dataset_y,min_components,max_components):
    """
    Performs PCA analysis and posts results to interactive webpage
    Inputs:
        Dataset
        Min Components
        Max Components
    Outputs:
        Pretty Pictures
    """
    #app = dash.Dash(__name__)
    app = JupyterDash(__name__)

    sl_min = min_components
    sl_max = max_components

    app.layout = html.Div([
        dcc.Graph(id="graph"),
        html.P("Number of components:"),
        dcc.Slider(
            id='slider',
            min=sl_min, max=sl_max, value=3,
            marks={i: str(i) for i in range(sl_min,sl_max+1)})
    ])

    @app.callback(
        Output("graph", "figure"), 
        [Input("slider", "value")])
    def run_and_plot(n_components):

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(dataset)

        var = pca.explained_variance_ratio_.sum() * 100

        labels = {str(i): f"PC {i+1}" 
                for i in range(n_components)}
        labels['color'] = 'bikes'

        fig = px.scatter_matrix(
            components,
            dimensions=range(n_components),
            labels=labels,
            color=dataset_y['bikes'],
            title=f'Total Explained Variance: {var:.2f}%')
        fig.update_traces(diagonal_visible=False)
        return fig

    app.run_server(mode = 'jupyterlab',port=3050)


def station_proximity(dataset):
    stations_lat = pd.unique(dataset['latitude'])
    stations_long = pd.unique(dataset['longitude'])

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

    def label_dist (row):
        x = near[int(row['station'])-int(201)]
        return x

    return dataset.apply (lambda row: label_dist(row), axis=1)
