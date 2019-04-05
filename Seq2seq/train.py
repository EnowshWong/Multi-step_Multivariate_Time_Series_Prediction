#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: Wong
# @Date  : 2019/3/27
# @Desc  : train
from Seq2seq.model import *
from Seq2seq.utils import *

input_seq_len = 31
output_seq_len = 31

total_iteractions = 1000
batch_size = 32
KEEP_RATE = 0.5

# 生成原始数据
X_train,y_train,X_test,y_test,y_mean,y_std = generate_my_data(output_seq_len)

# 生成输入和输出数据
x, y = generate_train_samples(X_train,y_train,input_seq_len,output_seq_len,batch_size)
print(x.shape, y.shape)

test_x, test_y = generate_test_samples(X_test,y_test,input_seq_len,output_seq_len)
print(test_x.shape, test_y.shape)

# 训练模型
train_losses = []
val_losses = []

rnn_model = build_graph(input_seq_len,output_seq_len,X_train.shape[1],y_train.shape[1],feed_previous=False)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Training losses: ")
    for i in range(total_iteractions):
        # 采样法，随机生成训练样本
        batch_input, batch_output = generate_train_samples(X_train,y_train,input_seq_len,output_seq_len,batch_size)

        # 输入3D Tensor(batch_size, seq_length, features_input) 赋值给enc_inp
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(input_seq_len)}
        # 输出Tensor，赋值给target_seq,update方法，把字典更新到feed_dict中
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(output_seq_len)})
        _, loss_t,train_preds = sess.run([rnn_model['train_op'], rnn_model['loss'],rnn_model['reshaped_outputs']], feed_dict)
        # np.expand_dims方法在维度1上添加pred数据
        train_preds = [np.expand_dims(pred, 1) for pred in train_preds]
        # np.concatenate 沿着维度1拼接数组值
        train_preds = np.concatenate(train_preds, axis=1)
        print('iter:',i,'loss:',loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'multivariate_ts_pollution_case'))

print("Checkpoint saved at: ", save_path)
