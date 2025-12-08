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
            df_train = pd.read_json('../../autodl-fs/Data/train.json') # ('./Data/train.json')
            print('加载 test.json')
            df_test = pd.read_json('../../autodl-fs/Data/test.json')
            print('加载 support.json')
            df_support = pd.read_json('../../autodl-fs/Data/support.json')
            print('加载 dev.json')
            df_dev = pd.read_json('../../autodl-fs/Data/dev.json')
            print('Finished')

            # 对原始数据集进行列筛选 - 只保留需要的列 , iloc是 pandas 中按位置索引选择数据的方法
            df_train = df_train.iloc[:,[0,1,2,3,5]] # 除domain之外的其余模块信息
            df_test = df_test.iloc[:,[0,1,2,3,5]]
            df_dev = df_dev.iloc[:,[0,1,2,3,5]]
            df_support = df_support.iloc[:,[0,1,2,3]]
            df_support['label'] = 'None' # 支持集没有标签信息

            # 拼接数据集,拼接后重新生成连续的索引
            self.df_data_labeled = pd.concat([df_train,df_dev,df_test],ignore_index=True) # 整合带标签的数据集
            self.df_data = pd.concat([df_train,df_dev,df_test,df_support],ignore_index=True) # 全部数据集

            self.save = save

    # 生成/加载label.pt（） - 所有user的标签信息
    def load_labels(self):
        print('加载 labels...', end=' ')
        path = os.path.join('./pt_data', 'label.pt')
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
    def build_graph(self):
        print('开始构建异质图',end='   ')
        path = os.path.join('./pt_data', 'edge_index.pt')
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
                os.makedirs('./pt_data', exist_ok=True)
                torch.save(edge_index, os.path.join('./pt_data', "edge_index.pt"))
                torch.save(edge_type, os.path.join('./pt_data', "edge_type.pt"))
        else:
            edge_index = torch.load(os.path.join('./pt_data', "edge_index.pt")).to(self.device)
            edge_type = torch.load(os.path.join('./pt_data', "edge_type.pt")).to(self.device)

        print('构建异质图完成')
        return edge_index, edge_type

    # 明确训练集、验证集、测试集在拼接后的标签数据中的索引范围
    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278,8278 + 2365)
        test_idx = range(8278+2365,8278+2365+1183)
        return train_idx,val_idx,test_idx


def create_dataloader(root='./processed_data', device='cuda', process=True, save=True):
    print('开始数据预处理...')

    # 确保必要的目录存在
    os.makedirs(root, exist_ok=True)
    os.makedirs('./pt_data', exist_ok=True)

    # 初始化数据集
    print('\n[1/6] 初始化 Twibot20 数据集...')
    dataset = Twibot20(root=root, device=device, process=process, save=save)

    # 加载标签数据
    print('\n[2/6] 加载标签数据...')
    labels = dataset.load_labels()
    print(f'✓ 标签数据加载完成，形状: {labels.shape}')

    # 处理用户简介
    print('\n[3/6] 处理用户简介数据...')
    descriptions = dataset.Des_preprocess()
    print(f'✓ 简介数据处理完成，数量: {len(descriptions)}')

    # 处理推文数据
    print('\n[4/6] 处理推文数据...')
    tweets = dataset.tweets_preprogress()
    print(f'✓ 推文数据处理完成，用户数量: {len(tweets)}')

    # 构建异质图
    print('\n[5/6] 构建异质图...')
    edge_index, edge_type = dataset.build_graph()
    print(f'✓ 异质图构建完成，边数量: {edge_index.shape[1]}, 边类型数量: {edge_type.shape[0]}')

    # 获取训练集、验证集、测试集的索引
    print('\n[6/6] 获取数据集划分索引...')
    train_idx, val_idx, test_idx = dataset.train_val_test_mask()
    print(f'✓ 训练集: {len(train_idx)} 样本')
    print(f'✓ 验证集: {len(val_idx)} 样本')
    print(f'✓ 测试集: {len(test_idx)} 样本')

    # 保存数据集信息摘要
    if save:
        print('\n保存数据集信息摘要...')
        summary = {
            'total_samples': len(labels),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
            'num_edges': edge_index.shape[1] if isinstance(edge_index, torch.Tensor) else 0,
            'description_samples': len(descriptions),
            'tweet_samples': len(tweets),
            'device': device,
            'created_at': dt.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        summary_path = os.path.join(root, 'dataset_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f'✓ 数据集摘要已保存至: {summary_path}')

    print('\n' + '='*60)
    print('数据加载器创建完成！')
    print('='*60)
    print(f'\n生成的文件位置:')
    print(f'  - 标签文件: ./pt_data/label.pt')
    print(f'  - 简介文件: {root}/description.npy')
    print(f'  - 推文文件: {root}/tweets.npy')
    print(f'  - 图结构文件: {root}/edge_index.pt, {root}/edge_type.pt')
    print(f'  - 数据摘要: {root}/dataset_summary.json')
    print('='*60 + '\n')


if __name__ == '__main__':
    # 当直接运行此脚本时，执行数据加载
    print('直接运行 dataset.py，开始加载和处理数据...\n')

    # 可以根据需要修改参数
    data = create_dataloader(
        root='./processed_data',
        device='cuda',  # 如果没有GPU，改为 'cpu'
        process=True,   # 如果已经处理过数据，可以设为 False
        save=True
    )

    print('数据加载完成！可以使用返回的 data 字典访问各类数据。')
    print(f'可用的数据键: {list(data.keys())}')


