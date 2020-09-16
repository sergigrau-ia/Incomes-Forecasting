import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Reshape, Dropout
from keras.models import Sequential
import numpy as np
from keras.layers import GaussianNoise as GN
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.core import Dropout
from keras.callbacks import LearningRateScheduler as LRS
from sklearn.model_selection import KFold
from keras.regularizers import l1
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import chart_studio.plotly as py
import plotly.graph_objs as go
# Offline mode",
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

def assignCOVIDFactor(df, threshold):
    if threshold == 'Pesimist':
        df.loc[(df['Date'] == "2020-03"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-04"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-05"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-06"), 'COVID'] = 1
        df.loc[(df['Date'] == "2020-07"), 'COVID'] = 1
        df.loc[(df['Date'] == "2020-08"), 'COVID'] = 2
        df.loc[(df['Date'] == "2020-09"), 'COVID'] = 3
        df.loc[(df['Date'] == "2020-10"), 'COVID'] = 3
        df.loc[(df['Date'] == "2020-11"), 'COVID'] = 4
        df.loc[(df['Date'] == "2020-12"), 'COVID'] = 4
    elif threshold == 'Minor_Optimist':
        df.loc[(df['Date'] == "2020-03"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-04"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-05"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-06"), 'COVID'] = 1
        df.loc[(df['Date'] == "2020-07"), 'COVID'] = 2
        df.loc[(df['Date'] == "2020-08"), 'COVID'] = 3
        df.loc[(df['Date'] == "2020-09"), 'COVID'] = 4
        df.loc[(df['Date'] == "2020-10"), 'COVID'] = 5
        df.loc[(df['Date'] == "2020-11"), 'COVID'] = 6
        df.loc[(df['Date'] == "2020-12"), 'COVID'] = 6
    elif threshold == 'Optimist':
        df.loc[(df['Date'] == "2020-03"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-04"), 'COVID'] = 0
        df.loc[(df['Date'] == "2020-05"), 'COVID'] = 1
        df.loc[(df['Date'] == "2020-06"), 'COVID'] = 2
        df.loc[(df['Date'] == "2020-07"), 'COVID'] = 3
        df.loc[(df['Date'] == "2020-08"), 'COVID'] = 4
        df.loc[(df['Date'] == "2020-09"), 'COVID'] = 5
        df.loc[(df['Date'] == "2020-10"), 'COVID'] = 6
        df.loc[(df['Date'] == "2020-11"), 'COVID'] = 6
        df.loc[(df['Date'] == "2020-12"), 'COVID'] = 6
    df = df.fillna(6)
    
    return df
        
def parseData(df, COVID, THRESHOLD_COVID):
    df['Date'] = pd.to_datetime(df['Date'])

    # Add Campaign column
    df.loc[(df['Date'].dt.month == 3) | (df['Date'].dt.month == 4), 'Campaign'] = 'Easter Campaign'
    df.loc[(df['Date'].dt.month == 7) | (df['Date'].dt.month == 8), 'Campaign'] = 'Summer Campaign'
    df.loc[df['Date'].dt.month == 10, 'Campaign'] = 'School Back Campaign'
    df.loc[(df['Date'].dt.month == 11) | (df['Date'].dt.month == 12), 'Campaign'] = 'Christmas Campaign'
    df = df.fillna('NORMAL')

    # Add special event column
    df.loc[(df['Date'].dt.year == 2015) | (df['Date'].dt.year == 2017) |
           (df['Date'].dt.year == 2018) | (df['Date'].dt.year == 2019), 'Special_Event'] = 'Yes'
    df = df.fillna('NO')

    # Add income sales same month a year ago
    if COVID:
        df['prev_sales_1'] = df['Incoming'].shift(1)
    else:
        df['prev_sales_1'] = df['Incoming'].shift(12)
    
    df = df.fillna(0)
    
    # Add COVID column
    if COVID:
        df = assignCOVIDFactor(df, THRESHOLD_COVID)
    
    # Add month column
    df['Month'] = df['Date'].dt.month
    
    # adding lags
    '''for inc in range(1, lag + 1):
        field_name = 'lag_' + str(inc)
        df[field_name] = df['Incoming'].shift(inc)'''

    # Encode categorical features
    encoder = ce.BinaryEncoder(cols=['Campaign', 'Special_Event', 'Month'])
    df_bin = encoder.fit_transform(df[['Campaign', 'Special_Event', 'Month']])
    df = df.drop(['Campaign', 'Special_Event', 'Month'], axis=1)
    df = pd.concat([df, df_bin], axis=1)
    
    #drop null values
    df = df.dropna().reset_index(drop=True)

    return df

def createSets(df, train_range, val_range, test_range, COVID):
    #apply Min Max Scaler    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(np.reshape(df['Incoming'].values, (-1, 1)))
    
    if COVID:
        scaler_covid = MinMaxScaler(feature_range=(0, 1))
        scaler_covid = scaler_covid.fit(np.reshape(df['COVID'].values, (-1, 1)))
        df['COVID'] = scaler_covid.transform(np.reshape(df['COVID'].values, (-1, 1)))
    
    df['Incoming'] = scaler.transform(np.reshape(df['Incoming'].values, (-1, 1)))
    df['prev_sales_1'] = scaler.transform(np.reshape(df['prev_sales_1'].values, (-1, 1)))      
            
    # Create masks
    train_mask = (df['Date'] >= train_range[0]) & (df['Date'] <= train_range[1])
    val_mask = (df['Date'] >= val_range[0]) & (df['Date'] <= val_range[1])
    test_mask = (df['Date'] >= test_range[0]) & (df['Date'] <= test_range[1])
    
    # Delete Date columns
    df = df.drop(['Date'], axis=1)

    # Create [train, val, test] sets.
    train_set = df.loc[train_mask].values
    val_set = df.loc[val_mask].values
    test_set = df.loc[test_mask].values   

    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])

    # reshape val set
    val_set = val_set.reshape(val_set.shape[0], val_set.shape[1])

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])

    # Get X's and Y's of all sets
    X_train, y_train = train_set[:, 1:], train_set[:, 0:1]
    X_test, y_test = test_set[:, 1:], test_set[:, 0:1]
    X_val, y_val = val_set[:, 1:], val_set[:, 0:1]

    print('x_train shape -> {}\n'.format(X_train.shape))
    print('y_train shape -> {}\n'.format(y_train.shape))

    print('x_val shape -> {}\n'.format(X_val.shape))
    print('y_val shape -> {}\n'.format(y_val.shape))

    print('x_test shape -> {}\n'.format(X_test.shape))
    print('y_test shape -> {}\n'.format(y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def deNormalize(normalized_list, min_value, max_value):
    denormalized_list = []
    for element in normalized_list:
      denormalized_list.append((element[0] * (max_value - min_value)) + min_value)
    return denormalized_list

def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)

def displaySeriesGraph(data):
    seas_d=sm.tsa.seasonal_decompose(data,model='add',freq=12);
    fig=seas_d.plot()
    fig.set_figheight(4)
    plt.show()

def selectBestArimaModel(x_values, y_values):
    step_wise=auto_arima(y_values, exogenous= x_values,
                         start_p=1, start_q=1, max_p=7, max_q=7,
                         d=1, max_d=7, trace=True, trend='t',
                         error_action='ignore', suppress_warnings=True,
                         stepwise=True)
    print(step_wise.summary())

def trainModel(y_train, X_train):
    # SARIMAX Model
    sxmodel = pm.auto_arima(y_train, exogenous=X_train,
                               start_p=1, start_q=1,
                               test='adf',
                               max_p=4, max_q=4, m=12,
                               start_P=0, seasonal=True,
                               d=None, D=1, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

    print(sxmodel.summary())
    sxmodel.fit(y_train, exogenous=X_train)
    return sxmodel

def deNormalize(normalized_list, min_value, max_value):
    denormalized_list = []
    for element in normalized_list:
      denormalized_list.append((element * (max_value - min_value)) + min_value)
    return denormalized_list

def displayResultsGraph(x, y, y_pred, COVID, path, threshold_covid):
    historic_data = go.Scatter(x=x,
                               y=y, 
                               name='Historic Data')
    
    predicted_data = go.Scatter(x=x,
                                y=y_pred, 
                                name='Predicted Data')
    
    layout = go.Layout(title='Puckator Sales trained COVID', xaxis=dict(title='Date'),
                       yaxis=dict(title='(Income Sales)'))
    
    fig = go.Figure(data=[historic_data, predicted_data], layout=layout)
    
    if COVID:
        fig.write_html('{}forecast_2020_covid_{}.html'.format(path, threshold_covid))
    else:
        fig.write_html('{}forecast_2020_final.html'.format(path))