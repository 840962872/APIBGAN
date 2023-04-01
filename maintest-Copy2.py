from math import sqrt

import numpy as np
import pandas as pd
import torch
from numpy import array
from pandas import DataFrame
from sklearn import preprocessing
from torch import nn
from torch.autograd import Variable

# 读取文件
data = pd.read_csv('yanfa_test_norm.csv',encoding="GB2312")

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(4, 256),  # 用线性变换将噪声映射到10个特征
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 10),  # 线性变换
        )

    def forward(self, x):
        x = self.gen(x)
        return x
for modeli in range(500):
    a = torch.load("model//generator_{}th.pth".format(modeli+1)).cuda()
    z = Variable(torch.randn(100, 4)).cuda()
    b = a(z).tolist()
    # print(b)
    data1 = DataFrame(b, columns=['day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret'])
    #b.to_csv("..//Generator_data.csv", encoding="GB2312")

    for i in range(len(data1['ret'])):
        #print(i)
        if data1['ret'][i] > 1 or data1['ret'][i] < 0:
            #print(data['ret'][i])
            data1.drop(axis=0, index=i, inplace=True)


    test_data = data[['day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret']]
    test_data = DataFrame(test_data, columns=['day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret'])

    '''min_max_scaler = preprocessing.MinMaxScaler()
    test_data = min_max_scaler.fit_transform(test_data)
    #print(train_data)
    test_data = DataFrame(test_data, columns=[ 'hour',
                              'account_IP', 'account_url', 'account_switchIP', 'account_url_IP',
                              'url_IP_switchIP', 'account_IP__count', 'account_url__count',
                                  'account_switchIP__count',
                              'account_url_IP__count','url_IP_switchIP__count','ret'])
    '''


    predict_list = []
    for diyi in range(len(test_data)):
        b = test_data.iloc[diyi].tolist()
        del (b[-1])
        dis_list = []
        for m in range(len(data1)):
            temp = data1.iloc[m].tolist()
            del (temp[-1])
            dis_list.append(np.sqrt(np.sum(np.square(array(b) - array(temp)))))
        # print(dis_list)
        # print(dis_list.index(min(dis_list)))
        # 打印每一个ret的预测值
        #print(data1.iloc[dis_list.index(min(dis_list))].tolist()[-1])
        predict_list.append(data1.iloc[dis_list.index(min(dis_list))].tolist()[-1])

    test_data.insert(10, 'predict_ret', predict_list)

    # 得分
    score = DataFrame((test_data['ret'] - test_data['predict_ret']) ** 2)
    print("这是第{}次的训练，最终得分为：".format(modeli+1))
    print(1 / (1 + sqrt(score.sum(axis=0) / score.count())))

    test_data.to_csv("gen_100_test_result//第{}th_score{}.csv".format(modeli+1, 1 / (1 + sqrt(score.sum(axis=0) / score.count()))),
                     index=None, encoding='GB2312')

