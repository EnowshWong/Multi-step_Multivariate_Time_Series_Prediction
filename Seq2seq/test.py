#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py.py
# @Author: Wong
# @Date  : 2019/3/27
# @Desc  : model test
from Seq2seq.model import *
from Seq2seq.utils import *

input_seq_len = 31
output_seq_len = 31
batch_size = 16

X_train,y_train,X_test,y_test,y_mean,y_std = generate_my_data(output_seq_len)


input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

x, y= generate_train_samples(X_train,y_train,input_seq_len,output_seq_len,batch_size)
test_x, test_y = generate_test_samples(X_test,y_test,input_seq_len,output_seq_len)

rnn_model = build_graph(input_seq_len,output_seq_len,input_dim,output_dim,feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    saver = rnn_model['saver']().restore(sess, os.path.join('./', 'multivariate_ts_pollution_case'))

    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)}  # batch prediction
    # 预测值初始化为0，赋值给target_seq
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in
                      range(output_seq_len)})
    # 得到预测值
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    # np.expand_dims方法在维度1上添加pred数据
    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    # np.concatenate 沿着维度1拼接数组值
    final_preds = np.concatenate(final_preds, axis=1)
    # 这里实际上输出了一整个月的值，final_preds
    print("Test mse is: ", np.mean((final_preds[:test_y.shape[0]] - test_y) ** 2))


final_preds_expand = [final_preds[i][-1][:] for i in range(final_preds.shape[0])]
final_preds_expand = np.array(final_preds_expand)

for i in range(final_preds_expand.shape[1]):
    final_preds_expand[:,i] = final_preds_expand[:,i] * y_std[i] + y_mean[i]
    y_test[:,i] = y_test[:,i] * y_std[i] + y_mean[i]
    y_train[:,i] = y_train[:,i] * y_std[i] + y_mean[i]

# 评价指标
mean_relative_error = 0.45 * np.mean(np.abs((final_preds_expand[:,0] - y_test[-output_seq_len:,0])) / y_test[-output_seq_len:,0]) + \
    0.55 * np.mean(np.abs((final_preds_expand[:,1] - y_test[-output_seq_len:,1])) / y_test[-output_seq_len:,1])

mean_square_error = 0.45 * np.mean(np.power(final_preds_expand[:,0] - y_test[-output_seq_len:,0],2)) + \
    0.55 * np.mean(np.power(final_preds_expand[:,1] - y_test[-output_seq_len:,1] / y_test[-output_seq_len:,1],2))
root_mean_square_error = 0.45 * np.sqrt(np.mean(np.power(final_preds_expand[:,0] - y_test[-output_seq_len:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power(final_preds_expand[:,1] - y_test[-output_seq_len:,1] / y_test[-output_seq_len:,1],2)))
mean_square_relative_error = 0.45 * np.mean(np.power((final_preds_expand[:,0] - y_test[-output_seq_len:,0]) / y_test[-output_seq_len:,0],2)) + \
    0.55 * np.mean(np.power((final_preds_expand[:,1] - y_test[-output_seq_len:,1]) / y_test[-output_seq_len:,1],2))
root_mean_square_relative_error = 0.45 * np.sqrt(np.mean(np.power((final_preds_expand[:,0] - y_test[-output_seq_len:,0]) / y_test[-output_seq_len:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power((final_preds_expand[:,1] - y_test[-output_seq_len:,1]) / y_test[-output_seq_len:,1],2)))
print('MRE:',mean_relative_error)
print('MSE:',mean_square_error)
print('RMSE:',root_mean_square_error)
print('MSRE:',mean_square_relative_error)
print('RMSRE:',root_mean_square_relative_error)

# with open('results.csv','w') as f:
#     f.writelines([str(mean_relative_error) + ' ',str(mean_square_error) + ' ',str(root_mean_square_error)+' ',str(root_mean_square_relative_error)+' '])


plt.subplot(211)
plt.plot(y_train[:,0],color = 'black',label = 'train actual')
plt.plot(range(y_train.shape[0],y_train.shape[0] + final_preds_expand.shape[0]),\
         final_preds_expand[:,0], color = 'orange', label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + final_preds_expand.shape[0]),\
         y_test[-output_seq_len:,0], color = 'blue', label = 'test actual')
plt.title('fitting result of purchase')
plt.legend()
plt.tight_layout()
plt.savefig('fitting result of purchase')

plt.subplot(212)
plt.plot(y_train[:,1],color = 'black',label = 'train actual')
plt.plot(range(y_train.shape[0],y_train.shape[0] + final_preds_expand.shape[0]),\
         final_preds_expand[:,1], color = 'orange', label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + final_preds_expand.shape[0]),\
         y_test[-output_seq_len:,1], color = 'blue', label = 'test actual')
plt.title('fitting result of redeem')
plt.legend()
plt.tight_layout()
plt.savefig('fitting result of redeem')
