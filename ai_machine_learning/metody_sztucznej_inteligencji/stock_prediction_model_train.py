"""
    Assumption: AI program to predict stock price of given company
"""

import math
import joblib
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path

PERCENTAGE = 0.75

plt.style.use('fivethirtyeight')

print(os.path)


def get_stock_data(company_symbol='AAPL', period=('2016-01-01', datetime.today().strftime("%Y-%m-%d"))):
    """
    :param company_symbol: symbol of company at the stock
    :param period: tuple, since when till when
    :return: pandas.DataFrame with stock information
    """
    return web.DataReader(company_symbol, data_source='yahoo', start=period[0], end=period[1])


# Taking out data to operate with NN from whole table
def extract_close_value(stock_data):
    """
    Extract Close price from stock information DataFrame
    :param stock_data: DataFrame stock information
    :return: numpy array with close price
    """
    data = stock_data.filter(['Close'])
    dataset = data.to_numpy()
    return dataset


def data_division(dataset, percentage):
    """
    Dividing the data into train and test packages
    :param dataset: numpy array dataset
    :param percentage: float value between 0 and 1 describing in what proportion data will be divided
    :return: numpy array train_data, numpy array test_data
    """
    training_data_len = math.ceil(len(dataset) * percentage)
    train_data = dataset[0:training_data_len]
    test_data = dataset[(training_data_len - 60):]
    return train_data, test_data


def sequential_model_creation(input_x, num_features):
    """
    Create a Keras Sequential
    :param input_x: integer, says how many elements in time series is
    :return: model: configured Sequential model
    """
    model = Sequential()
    model.add(LSTM(180, return_sequences=True, input_shape=(input_x, num_features)))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def plot_result(train, valid):
    """
    Plotting result with actual data
    :param train: DataFrame with data, on which model were trained
    :param valid: DataFrame with actual and predicted data for actual day
    """
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def main():
    time_stamp = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    path_to_models = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'))
    if not path_to_models.exists():
        path_to_models.mkdir(parents=True, exist_ok=True)

    company_name_shorts = ["AAPL", "AMZN", "FB", "MSFT",  "GOOG", "GOOGL", "CSCO", "INTC", "CMCSA", "PEP", "NFLX",
                           "ADBE", "^NDX"]

    stock_data2 = {company: get_stock_data(company, ('2012-01-01', datetime.now().strftime("%Y-%m-%d"))) for
                   company in company_name_shorts}


    # provide this same period for every company
    start_day = np.stack([val.first_valid_index() for key, val in stock_data2.items()]).max()
    dataset2 = np.stack([extract_close_value(val[start_day:]) for key, val in stock_data2.items()])

    scaler = MinMaxScaler()
    scaled_data = np.array([scaler.fit_transform(data) for data in dataset2])
    division_point = math.floor(scaled_data.shape[1] * PERCENTAGE)
    train_data = scaled_data[:, :division_point, :]
    test_data = scaled_data[:, (division_point-60):, :]

    inputs_train = train_data[:12]
    output_train = train_data[12]

    inputs_test = test_data[:12]
    output_test = test_data[12]

    x_train = np.array([[company[i - 60:i, 0] for i in range(60, len(company))] for company in inputs_train])
    y_train = np.array([output_train[i, 0] for i in range(60, len(output_train))])

    x_test = np.array([[company[i - 60:i, 0] for i in range(60, len(company))] for company in inputs_test])
    y_test = np.array([output_test[i, 0] for i in range(60, len(output_test))])

    x_train = x_train.transpose([1, 2, 0])
    x_test = x_test.transpose([1, 2, 0])


    ##########################################################################
    # stock_data = get_stock_data('AAPL', ('2012-01-01', datetime.now().strftime("%Y-%m-%d")))
    # dataset = extract_close_value(stock_data)
    # """
    #     add joblib.dump on data to restore them in prediction
    # """
    # # Normalisation
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(dataset)
    #
    # # Training data
    # train_data, test_data = data_division(scaled_data, PERCENTAGE)
    #
    # # Generating 60 days samples with possible prediction on next day for NN
    # # Train -> time series data
    # x_train = [train_data[i - 60:i, 0] for i in range(60, len(train_data))]
    # # Train -> day after 60 days sample
    # y_train = [train_data[i, 0] for i in range(60, len(train_data))]
    # # Test -> time series data
    # x_test = [test_data[i - 60:i, 0] for i in range(60, len(test_data))]
    # # Test -> day after 60 days sample
    # y_test = dataset[len(train_data):, :]
    #
    # # Some Reshapes
    # x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Creating model
    model = sequential_model_creation(x_train.shape[1], x_train.shape[2])

    model.fit(x_train, y_train, batch_size=1, epochs=3)

    model.save_weights(path_to_models / (time_stamp+".h5"))

    # Making predictions on model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(np.reshape(y_test, (y_test.shape[0], 1)))
    # Root mean squared error
    rmse = np.square(np.mean(predictions - y_test) ** 2)
    print(rmse)

    plot_data = stock_data2["^NDX"].filter(['Close'])[start_day:]
    # data = stock_data.filter(['Close'])
    train = plot_data[:train_data.shape[1]]
    valid = plot_data[train_data.shape[1]:]
    valid['Predictions'] = predictions

    plot_result(train, valid)


if __name__ == "__main__":
    main()
