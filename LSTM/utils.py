import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import numpy as np

def generate_data(seq_length):
    df = pd.read_excel('./data/data_features.xlsx')
    print(df.head())
    data = df.iloc[:,1:].copy().values
    x_train = data[:-seq_length, :].copy()
    # 将历史的预测目标数据转为当前特征
    y_train = data[:-seq_length, -2:].copy()  # 转换为二维tensor
    # test需要多取input_length个数据，以助于完整的预测
    x_test = data[-3 * seq_length + 1:, :].copy()
    y_test = data[-3 * seq_length + 1:, -2:].copy()

    # z-score处理，对one-hot编码不需要进行处理
    col_one_hot = [4, 5]
    for i in range(x_train.shape[1]):
        if i not in col_one_hot:
            mean = np.mean(x_train[:, i])
            std = np.std(x_train[:, i])
            x_train[:, i] = (x_train[:, i] - mean) / std
            x_test[:, i] = (x_test[:, i] - mean) / std

    y_mean: np.ndarray = np.array([])
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
    #生成3D Tensor(-1,seq_length,features)
    m = x_train.shape[0]

    x = np.zeros([m- 2 * seq_length + 1,seq_length,x_train.shape[1]])
    y = np.zeros([m - 2 * seq_length + 1,y_train.shape[1]]) # output_len = 1

    for i in range(x.shape[0]):
        x[i, :, :] = x_train[i:i + seq_length]
        y[i, :] = y_train[i + 2 * seq_length - 1]
    return x, y


def generate_test_samples(x_test,y_test,seq_length):
    # 生成3D Tensor(-1,features,seq_length)
    m = x_test.shape[0]

    x = np.zeros([seq_length , seq_length,x_test.shape[1]])
    y = np.zeros([seq_length, y_test.shape[1]])

    for i in range(x.shape[0]):
        x[i, :, :] = x_test[i:i + seq_length]
        y[i, :] = y_test[i + 2 * seq_length - 1]

    return x, y