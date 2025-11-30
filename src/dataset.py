import torch
import numpy as np
import pandas as pd
import json
import os

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
            df_train = pd.read_json('../../autodl-tmp/Data/train.json') # ('./Data/train.json')
            print('加载 test.json')
            df_test = pd.read_json('../../autodl-tmp/Data/test.json')
            # print('加载 support.json')
            # df_support = pd.read_json('../../autodl-fs/support.json')
            print('加载 dev.json')
            df_dev = pd.read_json('../../autodl-tmp/Data/dev.json')
            print('Finished')

            # 对原始数据集进行列筛选 - 只保留需要的列 , iloc是 pandas 中按位置索引选择数据的方法
            df_train = df_train.iloc[:,[0,1,2,3,5]] # 除domain之外的其余模块信息
            df_test = df_test.iloc[:,[0,1,2,3,5]]
            df_dev = df_dev.iloc[:,[0,1,2,3,5]]
            # df_support = df_support.iloc[:,[0,1,2,3]]
            # df_support['label'] = 'None' # 支持集没有标签信息

            # 拼接数据集,拼接后重新生成连续的索引
            self.df_data_labeled = pd.concat([df_train,df_dev,df_test],ignore_index=True) # 整合带标签的数据集
            # self.df_data = pd.concat([df_train,df_dev,df_test,df_support],ignore_index=True) # 全部数据集

            self.save = save

    # 生成/加载label.pt（） - 所有user的标签信息
    def load_labels(self):
        print('加载 labels...', end=' ')
        path = os.path.join(self.root, 'label.pt')
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled['label'].values).to(self.device)
            if self.save:
                # 确保目录存在
                os.makedirs(self.root, exist_ok=True)
                torch.save(labels, path)
        else:
            labels = torch.load(path).to(self.device)
        print('load_labels 完成')
        return labels

    # 用户简介处理 —— 这里可以看出，用户简介为空时，对应的填充值为None（联系专家系统应对特征不全的场景）
    # 为简介特征嵌入提供一种统一的输入vector<string>
    def Des_preprocess(self):
        print('加载简介description信息...',end = ' ')
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
                # 确保目录存在
                os.makedirs(self.root, exist_ok=True)
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True) # allow_pickle=True 允许加载包含字符串等 Python 对象的数组
        print('Des_preprocess finished')
        return description

    # 推文预处理，为推文信息嵌入提供统一的输入，和dec一个道理
    def tweets_preprogress(self):
        print('加载推文信息...',end = ' ')
        path = os.path.join(self.root, 'tweets.npy')
        if not os.path.exists(path):
            tweets = []
            for i in range(self.df_data_labeled.shape[0]):
                one_user_tweets = []
                if self.df_data_labeled['tweet'][i] is None:
                    one_user_tweets.append('')
                else:
                    for each in self.df_data_labeled['tweet'][i]:
                        one_user_tweets.append(each)
                tweets.append(one_user_tweets)
            if self.save:
                # 确保目录存在
                os.makedirs(self.root, exist_ok=True)
                # 保存变长列表为 object 数组，启用 allow_pickle
                tweets_obj = np.array(tweets, dtype=object)
                np.save(path, tweets_obj, allow_pickle=True)
        else:
            tweets = np.load(path, allow_pickle=True)
            # 将 object ndarray 转换为 Python 列表，便于后续处理
            try:
                tweets = tweets.tolist()
            except Exception:
                pass
        print('推文数据预处理完成')
        return tweets

    # 构建异质图
    def Build_Graph(self):
        print('开始构建异质图',end='   ')
        path = os.path.join(self.root, 'edge_index.pt')
        if not os.path.exists(path):
            id2index_dict = {id: index for index, id in enumerate(self.df_data['ID'])}
            edge_index = []
            edge_type = []
            for i, relation in enumerate(self.df_data['neighbor']):
                if relation is not None:
                    # following 关系 (edge_type = 0)
                    for each_id in relation['following']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                            edge_type.append(0)  # following 关系
                    # follower 关系 (edge_type = 1)
                    for each_id in relation['follower']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                            edge_type.append(1)  # follower 关系
                else:
                    continue
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
            
            if self.save:
                # 确保目录存在
                os.makedirs(self.root, exist_ok=True)
                torch.save(edge_index, os.path.join(self.root, "edge_index.pt"))
                torch.save(edge_type, os.path.join(self.root, "edge_type.pt"))
        else:
            edge_index = torch.load(os.path.join(self.root, "edge_index.pt")).to(self.device)
            edge_type = torch.load(os.path.join(self.root, "edge_type.pt")).to(self.device)
        
        print('构建异质图完成')
        return edge_index, edge_type

    # 明确训练集、验证集、测试集在拼接后的标签数据中的索引范围
    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278,8278 + 2365)
        test_idx = range(8278+2365,8278+2365+1183)
        return train_idx,val_idx,test_idx

