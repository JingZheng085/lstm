import paramiko
import pandas as pd
from datetime import datetime,timedelta
from holiday import public_holidays

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

filename='/mnt/mfs/work_zhjing/train_data.csv'

hostname = "192.168.16.23"
port = 32514
username = "zhjing"
password = "admin6688"
special_days=[datetime(2019,11,11),
              datetime(2019,12,25),
              datetime(2020,6,8),]

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


class training_data():
    def __init__(self,date):
        self.date=date
        self.yesterday=None
        self.server=connect_server()
        self.server.login()
        self.get_yesterday()

    def get_yesterday(self):
        yesterday=self.date- timedelta(days=1)

        while yesterday.weekday() in [5, 6]:
            yesterday -= timedelta(days=1)
            continue
        while yesterday in public_holidays:
            yesterday -= timedelta(days=1)
            continue
        while yesterday in special_days:
            yesterday -= timedelta(days=1)
            continue
        self.yesterday=yesterday

    def get_oi_data(self):
        path_oi='/media/hdd1/DAT_OPT/Tick/'+self.yesterday.strftime('%Y%m%d')




def get_yesterday()

def get_oi_data(date):





def main():

    server=connect_server(hostname,username,port,password)

    date=datetime.today()
    get_oi_data(date)

