#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : found_prediction.py
# @Author: Wong
# @Date  : 2019/2/15
# @Desc  : 预测2014年9月的赎回量

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel("./data.xlsx")

train = data[data['is_train'] == 1]
test = data[data['is_train'] == 0]
prediction = data[data['is_train'] == 2]

#训练的特征
features = data.columns[1:12]
purchase = data.columns[-2]

#预测申购量
clf = RandomForestRegressor(n_estimators= 100,oob_score=True,random_state= 42)

clf.fit(train[features],train[purchase])
preds = clf.predict(test[features])

#计算预测的误差
from sklearn.metrics import *
actual = test[purchase]
print("特征个数为10")
#计算相对误差和平均相对误差
relative_error = np.abs(actual-preds)/actual
ave_relative_error = np.mean(np.abs(actual - preds)/actual)
print("平均相对误差：",ave_relative_error)

#绘制预测图线与真实图线
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1,test.shape[0]+1),actual,'b.-')
plt.plot(range(1,test.shape[0]+1),preds,'r*-')
plt.title("comparison between the actual and the prediction")
plt.ylabel("amount")
plt.legend(['actual','prediction'])
plt.savefig('redeem')

#绘制权重图
plt.figure()
plt.bar(features,clf.feature_importances_)
plt.savefig('weights_redeem')