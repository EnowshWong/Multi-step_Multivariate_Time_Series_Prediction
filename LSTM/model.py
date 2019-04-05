#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: Wong
# @Date  : 2019/4/5
# @Desc  :
from LSTM.utils import *
import keras
import matplotlib.pyplot as plt
seq_length = 31
lstm_units = [50,40,50]
dense_units = 2
epochs = 50
batch_size = 32


x_train, y_train, x_test, y_test,y_mean,y_std = generate_data(seq_length)
# 3D Tensor (-1,seq_length,features)
train_x ,train_y = generate_train_samples(x_train,y_train,seq_length)
test_x,test_y = generate_test_samples(x_test,y_test,seq_length)

# LSTM的输入:(-1,time_step,features) 输出(-1,features)
# 构建3层LSTM，return_sequences = True，表示LSTM的输出为3D序列供下一层作为输入
model = keras.Sequential()
model.add(keras.layers.LSTM(lstm_units[0],input_shape=(train_x.shape[1],train_x.shape[2]),return_sequences=True))
model.add(keras.layers.LSTM(lstm_units[1],return_sequences=True))
model.add(keras.layers.LSTM(lstm_units[2]))
model.add(keras.layers.Dense(dense_units))
model.compile(loss='mse',optimizer='adam')
# train
# verbose: 日志显示, shuffle: 是否打乱数据集
history = model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,\
                    validation_data=(test_x,test_y),verbose=2,shuffle=False)

#predict
test_output = model.predict(test_x) #test_x(-1,seq_length,features)

for i in range(y_train.shape[1]):
    y_train[:,i] = y_train[:,i] * y_std[i] + y_mean[i]
    test_output[:,i] = test_output[:,i] * y_std[i] + y_mean[i]
    y_test[:,i] = y_test[:,i] * y_std[i] + y_mean[i]

plt.subplot(311)
plt.plot(y_train[:,0],label = 'train actual')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),test_output[:,0],color = 'red',label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),y_test[-seq_length:,0],color = 'green',label = 'test y')
plt.title('fitting result of purchase')
plt.tight_layout()
plt.legend()

plt.subplot(312)
plt.plot(y_train[:,1],label = 'train actual')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),test_output[:,1],color = 'red',label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),y_test[-seq_length:,1],color = 'green',label = 'test y')
plt.title('fitting result of redeem')
plt.tight_layout()
plt.legend()

#plot history
plt.subplot(313)
plt.plot(history.history['loss'],label = 'train loss')
plt.plot(history.history['val_loss'],label = 'test loss')
plt.title('loss')
plt.tight_layout()
plt.legend()

# 评价指标
mean_relative_error = 0.45 * np.mean(np.abs((test_output[:,0] - y_test[-seq_length:,0])) / y_test[-seq_length:,0]) + \
    0.55 * np.mean(np.abs((test_output[:,1] - y_test[-seq_length:,1])) / y_test[-seq_length:,1])

mean_square_error = 0.45 * np.mean(np.power(test_output[:,0] - y_test[-seq_length:,0],2)) + \
    0.55 * np.mean(np.power(test_output[:,1] - y_test[-seq_length:,1] / y_test[-seq_length:,1],2))
root_mean_square_error = 0.45 * np.sqrt(np.mean(np.power(test_output[:,0] - y_test[-seq_length:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power(test_output[:,1] - y_test[-seq_length:,1] / y_test[-seq_length:,1],2)))
mean_square_relative_error = 0.45 * np.mean(np.power((test_output[:,0] - y_test[-seq_length:,0]) / y_test[-seq_length:,0],2)) + \
    0.55 * np.mean(np.power((test_output[:,1] - y_test[-seq_length:,1]) / y_test[-seq_length:,1],2))
root_mean_square_relative_error = 0.45 * np.sqrt(np.mean(np.power((test_output[:,0] - y_test[-seq_length:,0]) / y_test[-seq_length:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power((test_output[:,1] - y_test[-seq_length:,1]) / y_test[-seq_length:,1],2)))
print('MRE:',mean_relative_error)
print('MSE:',mean_square_error)
print('RMSE:',root_mean_square_error)
print('MSRE:',mean_square_relative_error)
print('RMSRE:',root_mean_square_relative_error)