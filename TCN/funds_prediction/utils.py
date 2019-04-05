import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import numpy as np

def series_to_supervised(data, n_in, n_out,dropnan = True):
    # another way to transform time series prediction to supervised
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence:t-n,...,t-1
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]
    # forecast sequence
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' %(j+1,i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols,axis=1)
    agg.columns = names
    # drop rows with nans
    if dropnan:
        agg.dropna(inplace = True)
    return agg

def generate_data(seq_length):
    # seq_length is the input and output length
    df = pd.read_excel('../data/data_features.xlsx')
    print(df.head())
    data = df.iloc[:,1:].copy().values
    x_train = data[:-seq_length, :].copy()
    y_train = data[:-seq_length, -2:].copy()
    # predict pattern : x1,x2,...,xt -> yt+1,yt+2,...,yt+p
    # to direct predict rather than iterative predict, we lose 2*seq_length training data
    # using iterative strategy may cause error accumulation
    x_test = data[-3 * seq_length + 1:, :].copy()
    y_test = data[-3 * seq_length + 1:, -2:].copy()

    # z-score, especially one_hot vectors don't need
    col_one_hot = [4, 5]
    for i in range(x_train.shape[1]):
        if i not in col_one_hot:
            mean = np.mean(x_train[:, i])
            std = np.std(x_train[:, i])
            x_train[:, i] = (x_train[:, i] - mean) / std
            x_test[:, i] = (x_test[:, i] - mean) / std

    y_mean = np.array([])
    y_std = np.array([])
    for i in range(y_train.shape[1]):
        mean_y = np.mean(y_train[:, i])
        std_y = np.std(y_train[:, i])
        y_train[:, i] = (y_train[:, i] - mean_y) / std_y
        y_test[:, i] = (y_test[:, i] - mean_y) / std_y
        y_mean = np.append(y_mean, mean_y)
        y_std = np.append(y_std, std_y)

    return x_train, y_train, x_test, y_test,y_mean,y_std

def generate_train_samples(x_train,y_train,seq_length):
    #generate 3D Tensor(-1,features,seq_length)
    m = x_train.shape[0]

    # we use one step slipping window to generate samples
    # due to direct prediction, we lose 2 * seq_length training data
    x = torch.zeros([m- 2 * seq_length + 1,x_train.shape[1],seq_length])
    y = torch.zeros([m - 2 * seq_length + 1,y_train.shape[1],seq_length])

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    for i in range(x.shape[0]):
        x[i, :, :] = x_train[i:i + seq_length].transpose(1,0)
        y[i, :, :] = y_train[i + seq_length:i + 2 * seq_length].transpose(1,0)

    return Variable(x), Variable(y)


def generate_test_samples(x_test,y_test,seq_length):
    # generate 3D Tensor(-1,features,seq_length)
    m = x_test.shape[0]

    x = torch.zeros([seq_length , x_test.shape[1], seq_length])
    y = torch.zeros([seq_length, y_test.shape[1], seq_length])

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    for i in range(x.shape[0]):
        x[i, :, :] = x_test[i:i + seq_length].transpose(1, 0)
        y[i, :, :] = y_test[i + seq_length : i + 2 * seq_length].transpose(1, 0)

    return Variable(x), Variable(y)

def generate_predict_samples(x_predict,seq_length):
    m = x_predict.shape[0]

    x = torch.zeros([m - seq_length, x_predict.shape[1], seq_length])
    x_predict = torch.from_numpy(x_predict)
    for i in range(m - seq_length):
        x[i, :, :] = x_predict[i:i + seq_length].transpose(1, 0)

    return Variable(x)
