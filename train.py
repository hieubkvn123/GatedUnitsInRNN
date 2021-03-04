import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from network import Network
from datasets import get_data
from sklearn.preprocessing import MinMaxScaler

### Some constants ###
data_file = './data/stock.csv'
sequence_len = 12
test_size = 0.3
epochs = 15
learning_rate = 5e-2

ss = MinMaxScaler(feature_range=(-1, 1))
model = Network(rnn_cell='lstm', sigmoid=False, rnn_units=2)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
cosine_annealing = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=learning_rate)
X_train, X_test = get_data(data_file, sequence_len=sequence_len, test_size=test_size)


X_train = ss.fit_transform(X_train)
X_test  = ss.transform(X_test)
X_train = torch.Tensor(X_train).type(torch.FloatTensor)
X_test  = torch.Tensor(X_test).type(torch.FloatTensor)

def _make_sequence_prediction(model, inputs):
    outputs = []

    print('[INFO] Predicting ... ')
    with tqdm(total=len(inputs), file=sys.stdout, colour='green') as pbar:
        for i, input_ in enumerate(inputs):
            if (i == 0):
                outputs.append(input_)
            elif(i == len(inputs) - 1):
                break

            input_ = torch.FloatTensor(input_)
            output_, h = model(input_)
            outputs.append(output_)

        pbar.update(1)

    outputs = np.array(outputs).reshape(-1, 1)
    outputs = ss.inverse_transform(outputs)
    inputs = ss.inverse_transform(inputs)
    fig, ax = plt.subplots()
    ax.plot(outputs, color='blue', label='Predicted Stock Price')
    ax.plot(inputs, color='orange', label='Read Stock Price')
    ax.legend()

    plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
    plt.show()

def _validate(model, criterion, validation_data):
    model.eval()
    loss = 0.0
    losses = []

    print('[INFO] Validating ...')
    with tqdm(total = len(validation_data), file=sys.stdout, colour='green') as pbar:
        for j, seq in enumerate(validation_data):
            if(j == len(validation_data) - 1):
                break

            input_ = seq.reshape(1,1)
            target = validation_data[j+1].reshape(1,1)
            y, h = model(input_)
            y = y.reshape(1,1)

            # seq_loss = torch.sqrt(criterion(y, target))
            # mse = torch.pow((target - y), 2)
            # rmse = torch.sqrt(mse)
            seq_loss = criterion(y, target)

            loss += seq_loss
            losses.append(seq_loss.detach().numpy())
            pbar.update(1)

    return loss, np.array(losses).mean()

if(os.path.exists('checkpoint.pt')):
    print('[INFO] Loading checkpoint froom file ...')
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

torch.autograd.set_detect_anomaly(True)
model.train()
for i in range(epochs):
    losses = []
    running_loss = 0.0
    loss = 0.0

    #optimizer.zero_grad()
    with tqdm(total=len(X_train), file=sys.stdout) as pbar :
        for j, seq in enumerate(X_train):
            if(j == len(X_train) - 1):
                break

            input_ = seq.reshape(1,1)
            y, h = model(input_)
            y = y.reshape(1,1)
            target = X_train[j+1].reshape(1,1)
            # print(y, target, h)

            # mse = torch.pow((target - y), 2).sum() # criterion(y, target)
            # rmse = torch.sqrt(mse)
            seq_loss = criterion(y, target)

            losses.append(seq_loss.detach().numpy())
            loss += seq_loss

            pbar.update(1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.reset_hidden_state()
    val_loss, mean_val_loss = _validate(model, criterion, X_test)
    cosine_annealing.step()

    print('[INFO] Epoch #[%d/%d], MSE Loss = %.6f, Val MSE Loss = %.6f' % 
            (i+1, epochs, np.array(losses).mean(), mean_val_loss))

    if((i + 1) % 5 == 0):
        print('[INFO] Saving checkpoint to file ... ')
        torch.save({
            'model_state_dict' : model.state_dict(),
            'last_loss' : np.array(losses).mean()
        }, 'checkpoint.pt')

data = pd.read_csv(data_file, header=0).dropna()
data = data.sort_values('Date')

price_high = data['High'].values
price_low = data['Low'].values
price_mid = (price_high + price_low) / 2.0
price_mid = price_mid.reshape(-1, 1)
price_mid = ss.transform(price_mid)
_make_sequence_prediction(model, price_mid)
