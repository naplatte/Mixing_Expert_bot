import torch
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（必须在导入 src.model 之前）
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
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
            # print('加载 support.json')
            # df_support = pd.read_json('../../autodl-fs/Data/support.json')
            print('加载 dev.json')
            df_dev = pd.read_json('../../autodl-fs/Data/dev.json')
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
        path = os.path.join('../../autodl-fs/pt_data', 'label.pt')
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

    def num_prop_preprocess(self):
        print('开始处理数值类metadata', end='   ')
        path0 = os.path.join('../../autodl-fs/pt_data', 'num_properties_tensor.pt')
        if not os.path.exists(path0):
            if not os.path.exists(os.path.join('../../autodl-fs/pt_data', "followers_count.pt")):
                followers_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['followers_count'] is None:
                        followers_count.append(0)
                    else:
                        followers_count.append(self.df_data['profile'][i]['followers_count'])
                followers_count = torch.tensor(np.array(followers_count, dtype=np.float32)).to(self.device)
                if self.save:
                    os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                    torch.save(followers_count, os.path.join('../../autodl-fs/pt_data', "followers_count.pt"))

                friends_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['friends_count'] is None:
                        friends_count.append(0)
                    else:
                        friends_count.append(self.df_data['profile'][i]['friends_count'])
                friends_count = torch.tensor(np.array(friends_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(friends_count, os.path.join('../../autodl-fs/pt_data', 'friends_count.pt'))

                screen_name_length = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['screen_name'] is None:
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(len(self.df_data['profile'][i]['screen_name']))
                screen_name_length = torch.tensor(np.array(screen_name_length, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(screen_name_length, os.path.join('../../autodl-fs/pt_data', 'screen_name_length.pt'))

                favourites_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['favourites_count'] is None:
                        favourites_count.append(0)
                    else:
                        favourites_count.append(self.df_data['profile'][i]['favourites_count'])
                favourites_count = torch.tensor(np.array(favourites_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(favourites_count, os.path.join('../../autodl-fs/pt_data', 'favourites_count.pt'))

                active_days = []
                date0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['created_at'] is None:
                        active_days.append(0)
                    else:
                        date = dt.strptime(self.df_data['profile'][i]['created_at'], '%a %b %d %X %z %Y ')
                        active_days.append((date0 - date).days)
                active_days = torch.tensor(np.array(active_days, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(active_days, os.path.join('../../autodl-fs/pt_data', 'active_days.pt'))

                statuses_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['statuses_count'] is None:
                        statuses_count.append(0)
                    else:
                        statuses_count.append(int(self.df_data['profile'][i]['statuses_count']))
                statuses_count = torch.tensor(np.array(statuses_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(statuses_count, os.path.join('../../autodl-fs/pt_data', 'statuses_count.pt'))

            else:
                active_days = torch.load(os.path.join('../../autodl-fs/pt_data', "active_days.pt"))
                screen_name_length = torch.load(os.path.join('../../autodl-fs/pt_data', "screen_name_length.pt"))
                favourites_count = torch.load(os.path.join('../../autodl-fs/pt_data', "favourites_count.pt"))
                followers_count = torch.load(os.path.join('../../autodl-fs/pt_data', "followers_count.pt"))
                friends_count = torch.load(os.path.join('../../autodl-fs/pt_data', "friends_count.pt"))
                statuses_count = torch.load(os.path.join('../../autodl-fs/pt_data', "statuses_count.pt"))

            active_days = pd.Series(active_days.to('cpu').detach().numpy())
            active_days = (active_days - active_days.mean()) / active_days.std()
            active_days = torch.tensor(np.array(active_days))

            screen_name_length = pd.Series(screen_name_length.to('cpu').detach().numpy())
            screen_name_length_days = (screen_name_length - screen_name_length.mean()) / screen_name_length.std()
            screen_name_length_days = torch.tensor(np.array(screen_name_length_days))

            favourites_count = pd.Series(favourites_count.to('cpu').detach().numpy())
            favourites_count = (favourites_count - favourites_count.mean()) / favourites_count.std()
            favourites_count = torch.tensor(np.array(favourites_count))

            followers_count = pd.Series(followers_count.to('cpu').detach().numpy())
            followers_count = (followers_count - followers_count.mean()) / followers_count.std()
            followers_count = torch.tensor(np.array(followers_count))

            friends_count = pd.Series(friends_count.to('cpu').detach().numpy())
            friends_count = (friends_count - friends_count.mean()) / friends_count.std()
            friends_count = torch.tensor(np.array(friends_count))

            statuses_count = pd.Series(statuses_count.to('cpu').detach().numpy())
            statuses_count = (statuses_count - statuses_count.mean()) / statuses_count.std()
            statuses_count = torch.tensor(np.array(statuses_count))

            num_prop = torch.cat((followers_count.reshape([229580, 1]), friends_count.reshape([229580, 1]),
                                  favourites_count.reshape([229580, 1]), statuses_count.reshape([229580, 1]),
                                  screen_name_length_days.reshape([229580, 1]), active_days.reshape([229580, 1])),
                                 1).to(self.device)

            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(num_prop, os.path.join('../../autodl-fs/pt_data', "num_properties_tensor.pt"))

        else:
            num_prop = torch.load(os.path.join('../../autodl-fs/pt_data', "num_properties_tensor.pt")).to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('开始处理类别类metadata', end='   ')
        path = os.path.join('../../autodl-fs/pt_data', 'cat_properties_tensor.pt')
        if not os.path.exists(path):
            category_properties = []
            properties = ['protected', 'geo_enabled', 'verified', 'contributors_enabled', 'is_translator',
                          'is_translation_enabled', 'profile_background_tile', 'profile_use_background_image',
                          'has_extended_profile', 'default_profile', 'default_profile_image']
            for i in range(self.df_data.shape[0]):
                prop = []
                if self.df_data['profile'][i] is None:
                    for i in range(11):
                        prop.append(0)
                else:
                    for each in properties:
                        if self.df_data['profile'][i][each] is None:
                            prop.append(0)
                        else:
                            if self.df_data['profile'][i][each] == "True ":
                                prop.append(1)
                            else:
                                prop.append(0)
                prop = np.array(prop)
                category_properties.append(prop)
            category_properties = torch.tensor(np.array(category_properties, dtype=np.float32)).to(self.device)
            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(category_properties, os.path.join('../../autodl-fs/pt_data', 'cat_properties_tensor.pt'))
        else:
            category_properties = torch.load(os.path.join('../../autodl-fs/pt_data', "cat_properties_tensor.pt")).to(self.device)
        print('Finished')
        return category_properties

    # 构建异质图
    # def build_graph(self):
    #     print('开始构建异质图',end='   ')
    #     path = os.path.join('../../autodl-fs/pt_data', 'edge_index.pt')
    #     if not os.path.exists(path):
    #         id2index_dict = {id: index for index, id in enumerate(self.df_data['ID'])}
    #         edge_index = []
    #         edge_type = []
    #         for i, relation in enumerate(self.df_data['neighbor']):
    #             if relation is not None:
    #                 # following 关系 (edge_type = 0)
    #                 for each_id in relation['following']:
    #                     try:
    #                         target_id = id2index_dict[int(each_id)]
    #                     except KeyError:
    #                         continue
    #                     else:
    #                         edge_index.append([i, target_id])
    #                         edge_type.append(0)  # following 关系
    #                 # follower 关系 (edge_type = 1)
    #                 for each_id in relation['follower']:
    #                     try:
    #                         target_id = id2index_dict[int(each_id)]
    #                     except KeyError:
    #                         continue
    #                     else:
    #                         edge_index.append([i, target_id])
    #                         edge_type.append(1)  # follower 关系
    #             else:
    #                 continue
    #
    #         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
    #         edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
    #
    #         if self.save:
    #             # 确保目录存在
    #             os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
    #             torch.save(edge_index, os.path.join('../../autodl-fs/pt_data', "edge_index.pt"))
    #             torch.save(edge_type, os.path.join('../../autodl-fs/pt_data', "edge_type.pt"))
    #     else:
    #         edge_index = torch.load(os.path.join('../../autodl-fs/pt_data', "edge_index.pt")).to(self.device)
    #         edge_type = torch.load(os.path.join('../../autodl-fs/pt_data', "edge_type.pt")).to(self.device)
    #
    #     print('构建异质图完成')
    #     return edge_index, edge_type

    # 明确训练集、验证集、测试集在拼接后的标签数据中的索引范围
    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278,8278 + 2365)
        test_idx = range(8278+2365,8278+2365+1183)
        return train_idx,val_idx,test_idx


# def create_dataloader(root='./processed_data', device='cuda', process=True, save=True,
#                       use_mlp=True, output_dim=64, hidden_dim=512):
#     print('开始数据预处理...')
#
#     # 初始化数据集
#     print('\n[1/8] 初始化 Twibot20 数据集...')
#     dataset = Twibot20(root=root, device=device, process=process, save=save)
#
#
#     # 生成简介嵌入特征
#     print('\n[4/8] 生成简介嵌入特征...')
#     des_features = dataset.Des_embbeding(use_mlp=use_mlp, output_dim=output_dim, hidden_dim=hidden_dim)
#     print(f'✓ 简介嵌入特征生成完成，形状: {des_features.shape}')
#
#     # 生成推文嵌入特征
#     print('\n[6/8] 生成推文嵌入特征...')
#     tweets_features = dataset.tweets_embedding(use_mlp=use_mlp, output_dim=output_dim, hidden_dim=hidden_dim)
#     print(f'✓ 推文嵌入特征生成完成，形状: {tweets_features.shape}')
#
#
#     # 保存数据集信息摘要
#     if save:
#         print('\n保存数据集信息摘要...')
#         summary = {
#             'des_feature_dim': des_features.shape[1] if len(des_features.shape) > 1 else des_features.shape[0],
#             'tweets_feature_dim': tweets_features.shape[1] if len(tweets_features.shape) > 1 else tweets_features.shape[0],
#             'use_mlp': use_mlp,
#             'output_dim': output_dim if use_mlp else 1024,
#             'device': device,
#             'created_at': dt.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
#         summary_path = os.path.join(root, 'dataset_summary.json')
#         with open(summary_path, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, indent=4, ensure_ascii=False)
#         print(f'✓ 数据集摘要已保存至: {summary_path}')
#
#
# if __name__ == '__main__':
#     # 当直接运行此脚本时，执行数据加载
#     print('直接运行 dataset.py，开始加载和处理数据...\n')
#
#     # 可以根据需要修改参数
#     data = create_dataloader(
#         root='./processed_data',
#         device='cuda',  # 如果没有GPU，改为 'cpu'
#         process=True,   # 如果已经处理过数据，可以设为 False
#         save=True
#     )



