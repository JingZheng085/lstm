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
from statsmodels.formula.api import ols

neurons=[40,10,10,1]
d=0.01
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

def main():
    data=get_data(filename)
    train_data=data[:-100]

    model = ols("imp~oi_OTM+oi_ATM+call_vega+put_vega+call_gamma+put_gamma+spot_price"
                "+HistVol_30+HistVol_10+index_diff+eqt_higher_10days_mean", data=train_data).fit()
    test_data=data[-100:]
    pred=model.predict(test_data)
    plt.figure()
    plt.plot(test_data['imp'],label='actual')
    plt.plot(pred,label='pred')
    plt.legend(['actual','pred'])