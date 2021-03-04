import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data(data_file, sequence_len=12, test_size=0.4):
    data = pd.read_csv(data_file, header=0).dropna()
    data = data.sort_values('Date')

    price_high = data['High'].values
    price_low  = data['Low'].values
    price_mid  = (price_high + price_low) / 2.0

    data_len   = len(price_mid)
    num_sequence = data_len // sequence_len
    data_len   = num_sequence * sequence_len
    data = price_mid[:data_len]

    # data = data.reshape(num_sequence, sequence_len, 1)
    # labels = np.zeros((num_sequence, 1))

    # X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=test_size)
    num_train_sequence = int(data_len * (1 - test_size)) # int(num_sequence * test_size)
    num_test_sequence = data_len - num_train_sequence
    X_train = data[:num_train_sequence]
    X_test = data[num_train_sequence:]
    X_train = X_train.reshape(num_train_sequence, 1)
    X_test  = X_test.reshape(num_test_sequence, 1)
    data    = data.reshape(data_len, 1)


    return data, X_test

# get_data('../data/stock.csv')
