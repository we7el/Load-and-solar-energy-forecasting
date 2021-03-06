# LSTM model

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from matplotlib import pyplot
import numpy as np


'''
Returns X_train, X_test and y_train, y_test in the following format:
    X: array where each element is a week of data (example: 48 time steps * 7 days).
    y: array where each element is the corresponding next day (example: the next 48 time steps, i.e. the 8th day).
'''
def to_X_y(data, train, input_size, output_size):
    X_train, y_train, X_test, y_test = list(), list(), list(), list()
    for i in range(np.shape(data)[1]):
        input_start = 0
        for j in range(np.shape(data)[0]):
            input_end = input_start + input_size
            output_end = input_end + output_size
            if output_end < np.shape(data)[0]:
                if output_end < np.shape(train)[0]:
                    x_input = data[input_start:input_end, i]
                    x_input = x_input.reshape((len(x_input), 1))
                    X_train.append(x_input)
                    y_train.append(data[input_end:output_end, 0])
                else:
                    x_input = data[input_start:input_end, i]
                    x_input = x_input.reshape((len(x_input), 1))
                    X_test.append(x_input)
                    y_test.append(data[input_end:output_end, 0])
            input_start += output_size
    X_train = array(X_train)
    y_train = array(y_train)
    X_test = array(X_test)
    y_test = array(y_test)

    return X_train, y_train, X_test, y_test

'''
Returns the data split into train and test, X and y.
'''
def split_dataset_to_X_y(dataset, input_size, output_size):
    # Our data is over one month (~4 weeks), so our testing will be the last week
    training_split = 7 * output_size
    train, test = dataset[:-training_split], dataset[-training_split:]
    X_train, y_train, X_test, y_test = to_X_y(dataset, train, input_size, output_size)

    return X_train, y_train, X_test, y_test


'''
Returns Normalized Root Mean Squared Error for every time step as an array of size 48 and for all the time steps.
'''
def return_metrics(real, predicted):
    min_data = np.min(real)
    max_data = np.max(real)
    metrics = list()

    # NRMSE for one time step
    for i in range(real.shape[1]):
        MSE = mean_squared_error(real[:, i], predicted[:, i])
        RMSE = sqrt(MSE)
        metrics.append(RMSE)
    metrics = array(metrics) / (max_data - min_data)

    # NRMSE for all predicted
    tot = 0
    for row in range(real.shape[0]):
        for col in range(real.shape[1]):
            tot += (real[row, col] - predicted[row, col]) ** 2
    metric = sqrt(tot / (real.shape[0] * real.shape[1]))
    metric /= (max_data - min_data)

    return metric, metrics


'''
Returns the architecture of the model, i.e. non-trained model.
'''
def model_architecture(X_train, y_train):
    time_steps = X_train.shape[1]
    features = X_train.shape[2]
    output_size = y_train.shape[1]

    model = Sequential()
    model.add(LSTM(48, activation='sigmoid', return_sequences=True, input_shape=(time_steps, features)))
    model.add(LSTM(48, activation='sigmoid', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(output_size))
    model.compile(loss='mse', optimizer='adam')

    return model


'''
Returns trained model.
'''
def train_model(X_train, y_train, verbose=1, epochs=10, batch_size=100):
    model = model_architecture(X_train, y_train)
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.10, batch_size=batch_size, verbose=verbose)
    #model.save('simple_LSTM.h5')

    return model, history


'''
Plots the training loss and the validation loss.
'''
def plot_losses(h):
    pyplot.plot(h.history['loss'])
    pyplot.plot(h.history['val_loss'])
    pyplot.show()


'''
Returns an array of predictions. The array consists of 48 time step prediction for every "days_to_train_on" the model
is given.
The way the program is initially written is that, given 7 days worth of data, the model predicts the next day.
'''
def model_predict(model, X):
    predictions = list()
    for i in range(np.shape(X)[0]):
        x_input = X[i, :, :]
        x_input = np.reshape(x_input, (1, np.shape(x_input)[0], 1))
        y_pred = model.predict(x_input, verbose=0)
        predictions.append(y_pred[0])
    predictions = array(predictions)

    return predictions


'''
Loads, preprocesses data, trains the model and uses it to forecast and returns Normalized Root Mean Squared Error 
for every time step as an array of size 48 and for all the time steps.
all time steps.
'''
def forecast_model(dataset, input_size, output_size):
    X_train, y_train, X_test, y_test = split_dataset_to_X_y(dataset, input_size, output_size)
    model, h = train_model(X_train, y_train)
    plot_losses(h)
    predictions = model_predict(model, X_test)
    training_predictions = model_predict(model, X_train)
    metric, metrics = return_metrics(y_test, predictions)
    train_metric, train_metrics = return_metrics(y_train, training_predictions)


    return metric, metrics, train_metric, train_metrics, predictions

'''
Returns the parameters with which we kickstart the data loading, preprocessing and model training and forecasting
process.
'''
def start_params(str_csv):
    dataset = read_csv(str_csv, header=0, infer_datetime_format=True, parse_dates=['datetime'],
                   index_col=['datetime'])
    dataset = array(dataset)
    dataset = dataset.astype(np.float32)
    dataset = (dataset - np.mean(dataset)) / np.std(dataset)

    timesteps_per_day = 48
    days_to_train_on = 7
    days_to_output = 1
    input_size = timesteps_per_day * days_to_train_on
    output_size = timesteps_per_day * days_to_output

    return dataset, input_size, output_size

'''
Plots the predictions and the real data on the same figure for comparison and to display performance.
'''
def plot_predictions(predictions, dataset, input_size, output_size):
    X_train, y_train, X_test, y_test = split_dataset_to_X_y(dataset, input_size, output_size)
    pyplot.plot(y_test[0], label="Real")
    pyplot.plot(predictions[0], label="Predicted")
    pyplot.title("LSTM forecast")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    dataset, input_size, output_size = start_params('30_minute_data.csv')
    metric, metrics, train_metric, train_metrics, predictions = forecast_model(dataset, input_size, output_size)
    print(metric, train_metric)
    plot_predictions(predictions, dataset, input_size, output_size)