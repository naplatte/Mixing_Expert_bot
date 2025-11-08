from pydoc import describe
from xml.sax.handler import feature_external_pes

import torch
import numpy as np
import pandas as pd
import json
import os

from accelerate.commands.merge import description
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm

class Twibot20(Dataset):
    def __init__(self,root='./processed_data',device='cuda',process=True,save=True): # process:控制是否加载原始数据并进行数据预处理
        self.root = root
        self.device = device
        self.process = process
        if process:
            # 读取原始数据
            print('加载 train.json')
            df_train = pd.read_json('./Data/train.json')
            print('加载 test.json')
            df_test = pd.read_json('./Data/test.json')
            # print('加载 support.json')
            # df_support = safe_read_json('./Data/support.json')
            print('加载 dev.json')
            df_dev = pd.read_json('./Data/dev.json')
            print('Finished')

            # 对原始数据集进行列筛选 - 只保留需要的列 , iloc是 pandas 中按位置索引选择数据的方法
            df_train = df_train.iloc[:,[0,1,2,3,5]] # 除domain之外的其余模块信息
            df_test = df_test.iloc[:,[0,1,2,3,5]]
            df_dev = df_dev.iloc[:,[0,1,2,3,5]]
            # df_support = df_support.iloc[:,[0,1,2,3]]
            # df_support['label'] = 'None' # 支持集没有标签信息

            # 拼接数据集,拼接后重新生成连续的索引
            self.df_data_labeled = pd.concat([df_train,df_test,df_dev],ignore_index=True) # 整合带标签的数据集
            # 由于不使用 support.json，df_data_labeled 就是全量数据集
            self.df_data = self.df_data_labeled  # 为了兼容性，设置 df_data 等于 df_data_labeled
            self.save = save

    # 生成/加载label.pt（） - 所有user的标签信息
    def load_labels(self):
        print('加载 labels...',end=' ')
        path = self.root + 'label.pt'
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled['label']).to(self.device) # 提取label列转换为张量，可通过 .to() 方法在不同设备间移动：
            if self.save:
                torch.save(labels,'./processed_data/label.pt')
        else:
            labels = torch.load(self.root + "label.pt").to(self.device)
        print("finished")
        return labels

    # 用户简介处理 —— 这里可以看出，用户简介为空时，对应的填充值为None（联系专家系统应对特征不全的场景）
    # 为简介特征嵌入提供一种统一的输入vector<string>
    def Des_preprocess(self):
        print('加载简介description特征...',end = ' ')
        path = os.path.join(self.root, 'description.npy')
        if not os.path.exists(path):
            description = []
            for i in range(self.df_data_labeled.shape[0]):
                if self.df_data_labeled['profile'][i] is None or self.df_data_labeled['profile'][i]['description'] is None:
                    description.append('None')
                else:
                    description.append(self.df_data_labeled['profile'][i]['description'])
            description = np.array(description)
            if self.save:
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True) # allow_pickle=True 允许加载包含字符串等 Python 对象的数组
        print('finished')
        return description

    # 获取简介特征表示（嵌入）:因为其使用的是未经过微调的roberta，混合专家系统中需要使用微调的bert生成句向量
    # def Des_embedding(self):
    #     print('开始简介description特征嵌入')
    #     path = self.root + "des_tensor.pt"
    #     if not os.path.exists(path):
    #         description = np.load(os.path.join(self.root, 'description.npy'), allow_pickle=True)
    #         # 使用预训练语言模型获得嵌入表示 - distilroberta-base的隐藏层维度（即每个 token 的嵌入维度）为768 维
    #         print('加载 RoBerta')
    #         feature_extraction = pipeline('feature-extraction',model = "distilroberta-base",tokenizer="distilroberta-base",device=0) # 使用 Hugging Face 的 pipeline 创建 feature-extraction 任务处理器
    #         des_vec = []
    #         for each in tqdm(description):
    #             feature = torch.Tensor(feature_extraction(each))
    #             # 先累加一个句子中的所有词向量
    #             feature_tensor = None
    #             for (i, tensor) in enumerate(feature[0]):
    #                 if i == 0:
    #                     feature_tensor = tensor
    #                 else:
    #                     feature_tensor += tensor
    #
    #             # 对句子中所有词向量取平均（只计算一次）
    #             feature_tensor /= feature.shape[1]
    #             des_vec.append(feature_tensor)  # 每个句子只append一次
    #         des_tensor = torch.stack(des_vec,0).to(self.device)
    #         if self.save:
    #             torch.save(des_tensor,'./Data/des_tensor.pt')
    #     else:
    #         des_tensor = torch.load(self.root + "des_tensor.pt").to(self.device)
    #     print("finished")
    #     return des_tensor

    # 推文预处理，为推文信息嵌入提供统一的输入，和dec一个道理
    # def tweets_preprogress(self):

    # 获取推文特征表示（嵌入）
    # def tweets_embedding(self):
    #
    #
    # def num_prop_preprocess(self):
    #
    #
    #
    # def cat_prop_preprocess(self):
    #
    #
    # def Build_Graph(self):
    #
    #

    # 明确训练集、验证集、测试集在拼接后的标签数据中的索引范围
    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278,8278 + 2365)
        test_idx = range(8278+2365,8278+2365+1183)
        return train_idx,val_idx,test_idx

    # 先只关注description数据进行加载分析
    # def dataloader(self):
    #     labels = self.load_labels()
    #     if self.process:
    #         self.Des_preprocess()
    #         # self.tweets_preprocess()
    #     des_tensor = self.Des_embedding()
    #
    #     train_idx,val_idx,test_idx = self.train_val_test_mask()
    #     return des_tensor,train_idx,val_idx,test_idx