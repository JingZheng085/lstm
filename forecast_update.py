import paramiko
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import svm
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from numpy import concatenate

neurons=[40,10,10,1]
d=0.1
timesteps=7
split_ratio=0.2
filename='/mnt/mfs/work_zhjing/train_data.csv'



class connect_server():
    def __init__(self,hostname,username,port,password):
        self.hostname=hostname
        self.username=username
        self.port=port
        self.password=password
        self.client=paramiko.SSHClient()

    def login(self):
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.hostname, self.port, self.username, self.password, compress=True)
        self.sftp_client = self.client.open_sftp()

    def read_csv(self,filename):
        file=self.sftp_client.open(filename)
        data=pd.read_csv(file,index_col=0)
        file.close()
        return data

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def split_data(data,split_ratio,timesteps):
    n_features=int(len(data.columns)/(timesteps+1))
    values=data.values
    n_train_days=int(len(data)*split_ratio)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-n_features], train[:, -1]
    test_X, test_y = test[:, :-n_features], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))
    return train_X, train_y,test_X, test_y


def build_model(neurons,d,train_X):
    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dropout(d))
    model.add(Dense(neurons[1]))
    model.add(Dropout(d))
    model.add(Dense(neurons[2]))
    model.add(Dropout(d))
    model.add(Dense(neurons[3]))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
    return model

def model_score(model,X_train,y_train,X_test,y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    testScore = model.evaluate(X_test, y_test, verbose=0)
    #MSE
    return trainScore[0], testScore[0]



def measure_model(filename, d, neurons):
    df=get_data(filename)
    df_trans = scale_data(df)
    df_transformed = series_to_supervised(df_trans, timesteps, 1)
    X_train, y_train, X_test, y_test = split_data(df_transformed,split_ratio,timesteps)
    model = build_model( neurons, d,X_train)
    model.fit(X_train, y_train, batch_size=72, epochs=50, validation_split=0.1,
              verbose=1)

    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)
    return trainScore, testScore


def get_data(filename):
    hostname = "192.168.16.23"
    port = 32514
    username = "zhjing"
    password = "admin6688"
    server=connect_server(hostname,username,port,password)
    server.login()
    train_data=server.read_csv(filename)
    train_data=train_data.dropna()
    return train_data

def scale_data(data):
    scaler = MinMaxScaler()
    data_transformed=scaler.fit_transform(data)


    return data_transformed, scaler






def main():
    data=get_data(filename)

    data_transformed,scaler= scale_data(data)
    data_transformed= series_to_supervised(data_transformed, timesteps, 1)
    train_X, train_y,test_X, test_y=split_data(data_transformed, split_ratio, timesteps)

    model=build_model(neurons,d,train_X)
    # fit model
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    yhat = model.predict(test_X)

    mse=sqrt(mean_squared_error(test_y, yhat))
    y_pred=scaler.inverse_transform(yhat)
    y_actual=scaler.inverse_transform(concatenate(yhat, test_X[:,6],axis=1) )

    plt.figure()
    plt.plot(y_actual,label='actual')
    plt.plot(y_pred,label='pred')
    plt.legend(['actual','pred'])

