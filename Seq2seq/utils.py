#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Wong
# @Date  : 2019/3/27
# @Desc  :
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_my_data(seq_length):
    df = pd.read_excel('./data/data_features.xlsx')
    # print(df.head())
    data = df.iloc[:, 1:].copy().values
    x_train = data[:-seq_length, :].copy()
    y_train = data[:-seq_length, -2:].copy()  # 转换为二维tensor
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

    return x_train, y_train, x_test, y_test, y_mean, y_std


def generate_train_samples(x, y, input_seq_len,output_seq_len,batch_size=10):
    # 随机选择样本
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=False)

    input_batch_idxs = [list(range(i, i + input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq  # in shape: (batch_size, time_steps, feature_dim)


def generate_test_samples(x, y, input_seq_len, output_seq_len):
    # make sure that input_seq_len > output_seq_len
    total_samples = x.shape[0]

    input_batch_idxs = [list(range(i, i + input_seq_len)) for i in
                        range(input_seq_len)]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + input_seq_len, i + input_seq_len + output_seq_len)) for i in
                         range(output_seq_len)]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    # shape(-1,seq_length,features)
    return input_seq, output_seq
