import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime as dt
from torch.utils.data import Dataset


class Twibot20(Dataset):
    def __init__(self, root='./processed_data', device='cuda', process=True, save=True):
        self.root = root
        self.device = device
        self.process = process
        self.save = save
        if process:
            print('加载 train.json')
            df_train = pd.read_json('../../autodl-fs/Data/train.json')
            print('加载 test.json')
            df_test = pd.read_json('../../autodl-fs/Data/test.json')
            print('加载 support.json')
            df_support = pd.read_json('../../autodl-fs/Data/support.json')
            print('加载 dev.json')
            df_dev = pd.read_json('../../autodl-fs/Data/dev.json')
            print('Finished')

            df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
            df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]
            df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]
            df_support = df_support.iloc[:, [0, 1, 2, 3]]
            df_support['label'] = 'None'

            self.df_data_labeled = pd.concat([df_train, df_dev, df_test], ignore_index=True)
            self.df_data = pd.concat([df_train, df_dev, df_test, df_support], ignore_index=True)

    def load_labels(self):
        print('加载 labels...', end=' ')
        path = os.path.join('../../autodl-fs/pt_data', 'label.pt')
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled['label'].values).to(self.device)
            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(labels, path)
        else:
            labels = torch.load(path).to(self.device)
        print('load_labels 完成')
        return labels

    def Des_preprocess(self):
        print('加载简介description信息...', end=' ')
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
                os.makedirs(self.root, exist_ok=True)
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True)
        print('Des_preprocess finished')
        return description

    def tweets_preprogress(self):
        print('加载推文信息...', end=' ')
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
                os.makedirs(self.root, exist_ok=True)
                np.save(path, np.array(tweets, dtype=object), allow_pickle=True)
        else:
            tweets = np.load(path, allow_pickle=True)
            try:
                tweets = tweets.tolist()
            except:
                pass
        print('推文数据预处理完成')
        return tweets

    def num_prop_preprocess(self, all_nodes=False):
        """处理数值型元数据 - 5个属性: followers/friends/statuses_count, screen_name_length, active_days"""
        print('开始处理数值类metadata', end='   ')
        suffix = '_all' if all_nodes else '_labeled'
        path = os.path.join('../../autodl-fs/pt_data', f'num_properties{suffix}_tensor.pt')

        if not os.path.exists(path):
            df = self.df_data if all_nodes else self.df_data_labeled
            n = df.shape[0]

            def safe_int(val):
                try:
                    return int(float(val)) if val is not None else 0
                except (ValueError, TypeError):
                    return 0

            followers, friends, screen_len, statuses, active = [], [], [], [], []
            date0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')

            for i in range(n):
                p = df['profile'][i]
                if p is None:
                    followers.append(0); friends.append(0); screen_len.append(0)
                    statuses.append(0); active.append(0)
                else:
                    followers.append(safe_int(p.get('followers_count')))
                    friends.append(safe_int(p.get('friends_count')))
                    screen_len.append(len(str(p.get('screen_name') or '')))
                    statuses.append(safe_int(p.get('statuses_count')))
                    try:
                        date = dt.strptime(p['created_at'], '%a %b %d %X %z %Y ')
                        active.append((date0 - date).days)
                    except:
                        active.append(0)

            def zscore(arr):
                arr = np.array(arr, dtype=np.float32)
                mean = np.mean(arr)
                std = np.std(arr)
                if std == 0:
                    return torch.zeros(len(arr), dtype=torch.float32)
                return torch.tensor((arr - mean) / std, dtype=torch.float32)

            num_prop = torch.stack([
                zscore(followers), zscore(friends), zscore(statuses),
                zscore(screen_len), zscore(active)
            ], dim=1).to(self.device)

            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(num_prop, path)
        else:
            num_prop = torch.load(path).to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self, all_nodes=False):
        """处理类别型元数据 - 3个属性: 是否私密、是否认证、是否默认头像"""
        print('开始处理类别类metadata', end='   ')
        suffix = '_all' if all_nodes else '_labeled'
        path = os.path.join('../../autodl-fs/pt_data', f'cat_properties{suffix}_tensor.pt')

        if not os.path.exists(path):
            df = self.df_data if all_nodes else self.df_data_labeled
            n = df.shape[0]
            cat_props = []

            for i in range(n):
                p = df['profile'][i]
                if p is None:
                    cat_props.append([0, 0, 1])
                else:
                    protected = 1 if p.get('protected') == "True " else 0
                    verified = 1 if p.get('verified') == "True " else 0
                    img_url = p.get('profile_image_url')
                    default_img = 1 if (img_url is None or img_url == '' or
                        img_url == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png') else 0
                    cat_props.append([protected, verified, default_img])

            cat_tensor = torch.tensor(cat_props, dtype=torch.float32).to(self.device)
            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(cat_tensor, path)
        else:
            cat_tensor = torch.load(path).to(self.device)
        print('Finished')
        return cat_tensor

    def build_graph(self):
        print('开始构建异质图', end='   ')
        path = os.path.join('../../autodl-fs/pt_data', 'edge_index.pt')
        if not os.path.exists(path):
            id2index_dict = {id: index for index, id in enumerate(self.df_data['ID'])}
            edge_index, edge_type = [], []
            for i, relation in enumerate(self.df_data['neighbor']):
                if relation is not None:
                    for each_id in relation['following']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                            edge_index.append([i, target_id])
                            edge_type.append(0)
                        except KeyError:
                            continue
                    for each_id in relation['follower']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                            edge_index.append([i, target_id])
                            edge_type.append(1)
                        except KeyError:
                            continue
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
            if self.save:
                os.makedirs('../../autodl-fs/pt_data', exist_ok=True)
                torch.save(edge_index, os.path.join('../../autodl-fs/pt_data', "edge_index.pt"))
                torch.save(edge_type, os.path.join('../../autodl-fs/pt_data', "edge_type.pt"))
        else:
            edge_index = torch.load(os.path.join('../../autodl-fs/pt_data', "edge_index.pt")).to(self.device)
            edge_type = torch.load(os.path.join('../../autodl-fs/pt_data', "edge_type.pt")).to(self.device)
        print('构建异质图完成')
        return edge_index, edge_type

    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278, 8278 + 2365)
        test_idx = range(8278 + 2365, 8278 + 2365 + 1183)
        return train_idx, val_idx, test_idx

