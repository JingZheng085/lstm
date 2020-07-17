import paramiko
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import svm
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras import optimizers
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from datetime import datetime


neurons=[40,10,10,1]
d=0.2
timesteps=7
split_ratio=0.8
filename='/mnt/mfs/dat_zhjing/300_500/daily_300_500/training_data.csv'



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
    n_features=int(len(data.columns)/(2*timesteps))
    values=data.values
    n_train_days=int(len(data)*split_ratio)

    train = values[:n_train_days, :]
    test = values[n_train_days:-1, :]
    pred=values[-1, :]
    # split into input and outputs
    train_X, train_y = train[:, :n_features*timesteps], train[:, -n_features*timesteps:]
    test_X, test_y = test[:, :n_features*timesteps], test[:, -n_features*timesteps:]

    pred_X=pred[ :n_features*timesteps]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))
    pred_X = pred_X.reshape((1, timesteps, n_features))
    return train_X, train_y,test_X, test_y,pred_X


def build_model(neurons,d,train_X):
    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dropout(d))
    model.add(Dense(neurons[1]))
    model.add(Dropout(d))
    model.add(Dense(neurons[2]))
    model.add(Dropout(d))
    model.add(Dense(neurons[3]))
    # adam = optimizers.Adam(decay=0.2)
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
    # data_X=data.drop('diff',axis=1)
    # data_Y=data['diff']
    data_transformed=scaler.fit_transform(data)
    # features_n = len(data.columns)
    # data_transformed=np.insert(data_X_transformed,features_n-1,data_Y/1000,axis=1)

    return data_transformed


def inverse_scale(data_y,pred_y):
    data_y=data_y.values.reshape(-1,1)
    pred_y=pred_y.reshape(-1,1)
    scaler = MinMaxScaler()
    transformed_y= scaler.fit_transform(data_y)
    new_y=scaler.inverse_transform(pred_y)
    return new_y





def main():

    data=get_data(filename)
    data=data.apply(lambda x:x.replace(np.inf, 1) )
    data = data.apply(lambda x: x.replace(-np.inf, -1))
    today=datetime.today()
    data.loc[today.strftime('%Y-%m-%d')]=0



    data.apply(lambda x: x.shift(1))
    data['diff']=data['diff'].shift(-1)
    data.dropna(inplace=True)
    data_transformed= scale_data(data)
    data_transformed = series_to_supervised(data_transformed, timesteps, timesteps)
    # train_X, train_y,pred_X, pred_y=split_data(data_transformed, split_ratio, timesteps)
    train_X, train_y, test_X,test_y,pred_X = split_data(data_transformed, split_ratio,timesteps)

    model=build_model(neurons,d,train_X)
    # fit model

    history = model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    yhat = model.predict(pred_X)
    pred_y=inverse_scale(data['diff'], yhat )
    # mse=sqrt(mean_squared_error(test_y, yhat))

    a=model.predict(test_X)


    plt.figure()
    plt.plot(inverse_scale(data['diff'], test_y ),label='actual')
    plt.plot(inverse_scale(data['diff'], a ),label='pred')
    plt.legend(['actual','pred'])


