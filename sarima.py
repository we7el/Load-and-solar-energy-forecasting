# SARIMA model

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sampling import Resampling
from pyramid.arima import auto_arima
from pandas import read_csv
from numpy import array
from numpy import reshape
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


'''
Returns train data and test data.
'''
def preprocess_data(str_csv):
    dataset = read_csv(str_csv, header=0, infer_datetime_format=True,
                   parse_dates=['datetime'], index_col=['datetime'])
    dataset = array(dataset)
    dataset = dataset[:, 0]
    data_raw = reshape(dataset, (len(dataset), 1))
    data = data_raw[:-48]
    test = data_raw[-48:]

    return data, test

'''
Trains the SARIMA model and returns the trained model.
'''
def train_SARIMA(data):
    my_order = (1, 0, 1)
    my_seasonal_order = (1, 0, 3, 48)
    model = SARIMAX(data, order=my_order, seasonal_order=my_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(maxiter=5)

    return model_fit

'''
Plots yhat and test on the same figure for comparison.
'''
def plot_prediction(yhat, test):
    pyplot.plot(yhat, label="Forecast")
    pyplot.plot(test, label="Real")
    pyplot.legend()
    pyplot.title("Load forecast SARIMA model")
    pyplot.show()

'''
Returns the Root mean squared error (RMSE) and the Normalized RMSE (NRMSE) of the forecasting
'''
def get_metrics(yhat, test):
    mse = mean_squared_error(test, yhat)
    rmse = sqrt(mse)
    max_data = np.max(test)
    min_data = np.min(test)
    nrmse = rmse / (max_data - min_data)

    return rmse, nrmse

'''
Loads and preprocesses the data, trains the model, and uses it to predict a sequence of the loads.
'''
def forecast_SARIMA(steps_forecasted=48):
    data, test = preprocess_data('30_minute_data.csv')
    model_fit = train_SARIMA(data)
    yhat = model_fit.forecast(steps_forecasted)
    yhat = np.reshape(yhat, (steps_forecasted,1))
    rmse, nrmse = get_metrics(yhat, test)
    plot_prediction(yhat, test)

    return rmse, nrmse


if __name__ == '__main__':
    rmse, nrmse = forecast_SARIMA()
    print(rmse, nrmse)
    print('program ended')