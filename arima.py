import pandas as pd
from lib.utils import parseData, createSets, trainModel, deNormalize, displayResultsGraph
from sklearn.metrics import mean_squared_error
import numpy as np

THRESHOLD_VALUES = ['Pesimist', 'Minor_Optimist', 'Optimist']

COVID = False
THRESHOLD_COVID = THRESHOLD_VALUES[2]
FORECASTS_PATH = 'forecasts/'

if COVID:
    TRAIN_RANGE = ['2017-01-01', '2018-12-31']
    VAL_RANGE = ['2019-01-01', '2020-05-31']
    TEST_RANGE = ['2020-06-01', '2020-12-31']
    N_PERIODS = 1
else:
    TRAIN_RANGE = ['2017-01-01', '2018-12-31']
    VAL_RANGE = ['2019-01-01', '2019-10-31']
    TEST_RANGE = ['2019-11-01', '2020-12-31']
    N_PERIODS = 14

if __name__ == "__main__":
    # Get the dates range
    train_range = pd.date_range(start=TRAIN_RANGE[0], end=TRAIN_RANGE[1], freq='M')
    
    val_range = pd.date_range(start=VAL_RANGE[0], end=VAL_RANGE[1], freq='M')
    
    if COVID:
        test_range = ["2020-06-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01", 
                      "2020-12-01"]
    else:
        test_range = pd.date_range(start=TEST_RANGE[0], end=TEST_RANGE[1], freq='M')
    
        
    # Read data
    df = pd.read_csv('historical_data.csv', sep=';')
    original_df = df.copy()
    
    # Transform data
    df = parseData(df, COVID, THRESHOLD_COVID)
    
    # Get cluster labels
   
    # Create Sets    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = createSets(df, TRAIN_RANGE, VAL_RANGE, TEST_RANGE, COVID)
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    
    # Train the model
    sxmodel = trainModel(y_train, X_train)
    
    # Get predictions for a period            
    if COVID:
        predicted_values = []
        isFirst = True
        for index, element in enumerate(X_test):
            if isFirst:
                fitted, confint = sxmodel.predict(n_periods=N_PERIODS, 
                                                  exogenous=np.reshape(element, (1, len(element))), 
                                                  return_conf_int=True)
                predicted_values.append(fitted[0])
                isFirst = False
            else:
                element[0] = fitted[0]
                fitted, confint = sxmodel.predict(n_periods=N_PERIODS, 
                                                  exogenous=np.reshape(element, (1, len(element))), 
                                                  return_conf_int=True)
                predicted_values.append(fitted[0])
            X_train = np.concatenate((X_train, np.reshape(element, (1, len(element)))), axis=0)
            y_train = np.concatenate((y_train, np.reshape(fitted[0], (1, 1))), axis=0)
            
            sxmodel.fit(y_train, exogenous=X_train)
    else:
        fitted, confint = sxmodel.predict(n_periods=N_PERIODS, 
                                          exogenous=X_test, 
                                          return_conf_int=True)
    
    # Denormalize the numerical columns
    if COVID:
        fitted_normal = scaler.inverse_transform(np.reshape(predicted_values, (-1, 1))).tolist()
    else:
        fitted_normal = scaler.inverse_transform(np.reshape(fitted, (-1, 1))).tolist()
    
    actual_normal = scaler.inverse_transform(np.reshape(y_test, (-1, 1))).tolist()   
    val_normal = scaler.inverse_transform(np.reshape(y_val, (-1, 1))).tolist()   
    
    # Transform list of lists to list of values
    fitted_normal = [y for x in fitted_normal for y in x]
    actual_normal = [y for x in actual_normal for y in x]
    val_normal = [y for x in val_normal for y in x]
 
    # Obtain the mean squared error
    error = mean_squared_error(actual_normal, fitted_normal)
    print('Mean Squared Error -> {}\n'.format(error))
      
    # Display results on a graph
    displayResultsGraph(test_range, actual_normal, fitted_normal, COVID, 
                        FORECASTS_PATH, THRESHOLD_COVID)
    
   