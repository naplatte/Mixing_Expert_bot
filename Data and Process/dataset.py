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

try:
    from src.model import MLP
except ImportError as e:
    print(f"警告: 无法导入 MLP 类: {e}")
    print(f"当前 Python 路径: {sys.path}")
    print(f"项目根目录: {project_root}")
    raise

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

    def Des_embbeding(self, use_mlp=True, output_dim=64, hidden_dim=512):
        print('开始des数据嵌入 (使用 BAAI/bge-m3 模型)')

        # 根据是否降维选择保存路径
        if use_mlp:
            path = os.path.join('./pt_data', f"des_feature_{output_dim}d.pt")
            mlp_path = os.path.join('./pt_data', f"des_mlp_{output_dim}d.pt")
        else:
            path = os.path.join('./pt_data', "des_feature_1024d.pt")

        if not os.path.exists(path):
            description = np.load(os.path.join(self.root, 'description.npy'), allow_pickle=True)
            print('加载 BAAI/bge-m3 模型...')

            # 使用 SentenceTransformer 加载 BGE-M3 模型
            model_name = "BAAI/bge-m3"
            model = SentenceTransformer(model_name)

            # 将模型移到对应设备
            device = self.device if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            model = model.to(device)

            des_vec = []
            print('开始提取特征...')

            # 批量处理以提高效率
            batch_size = 32
            for i in tqdm(range(0, len(description), batch_size)):
                batch_texts = []
                for j in range(i, min(i + batch_size, len(description))):
                    # 清理文本
                    text = str(description[j]).strip()
                    if text == '' or text.lower() == 'none':
                        text = ''  # 空简介用空字符串
                    batch_texts.append(text)

                # 使用 BGE-M3 模型进行编码
                with torch.no_grad():
                    embeddings = model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=device,
                        normalize_embeddings=True
                    )

                    for emb in embeddings:
                        des_vec.append(emb.cpu())

            # 堆叠成张量 [num_users, 1024]
            des_tensor = torch.stack(des_vec, 0)
            print(f'BGE-M3 特征提取完成. Shape: {des_tensor.shape}')

            # 使用 MLP 降维
            if use_mlp:
                print(f'使用 MLP 降维到 {output_dim} 维...')

                mlp = MLP(input_size=1024, output_size=output_dim, hidden_size=hidden_dim).to(device)
                mlp.eval()

                with torch.no_grad():
                    des_tensor = mlp(des_tensor.to(device)).cpu()

                print(f'降维完成. Shape: {des_tensor.shape}')

            des_tensor = des_tensor.to(self.device)

            if self.save:
                os.makedirs('./pt_data', exist_ok=True)
                torch.save(des_tensor, path)
                print(f'特征已保存至 {path}')
        else:
            print(f'从 {path} 加载预计算的特征')
            des_tensor = torch.load(path).to(self.device)

        print(f'Des_embbeding 完成. Shape: {des_tensor.shape}')
        return des_tensor

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

    # 推文数据嵌入
    def tweets_embedding(self, use_mlp=True, output_dim=64, hidden_dim=512):
        print('开始推文数据嵌入 (使用 BAAI/bge-m3 模型)')

        # 根据是否降维选择保存路径
        if use_mlp:
            path = os.path.join('./pt_data', f"tweets_feature_{output_dim}d.pt")
            mlp_path = os.path.join('./pt_data', f"tweets_mlp_{output_dim}d.pt")
        else:
            path = os.path.join('./pt_data', "tweets_feature_1024d.pt")

        if not os.path.exists(path):
            tweets = np.load(os.path.join(self.root, 'tweets.npy'), allow_pickle=True)
            print('加载 BAAI/bge-m3 模型...')

            # 使用 SentenceTransformer 加载 BGE-M3 模型
            model_name = "BAAI/bge-m3"
            model = SentenceTransformer(model_name)

            # 将模型移到对应设备
            device = self.device if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            model = model.to(device)

            tweets_list = []
            print('开始提取推文特征...')

            # 遍历每个用户的推文
            for each_person_tweets in tqdm(tweets):
                if len(each_person_tweets) == 0 or (len(each_person_tweets) == 1 and each_person_tweets[0] == ''):
                    # 处理空推文的情况：编码一个空字符串
                    with torch.no_grad():
                        user_embedding = model.encode(
                            [''],
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_tensor=True,
                            device=device,
                            normalize_embeddings=True
                        )
                    tweets_list.append(user_embedding.squeeze(0).cpu())
                else:
                    # 清理推文文本
                    cleaned_tweets = []
                    for tweet in each_person_tweets:
                        text = str(tweet).strip()
                        if text == '':
                            text = ''
                        cleaned_tweets.append(text)

                    # 批量编码该用户的所有推文
                    with torch.no_grad():
                        tweet_embeddings = model.encode(
                            cleaned_tweets,
                            batch_size=32,
                            show_progress_bar=False,
                            convert_to_tensor=True,
                            device=device,
                            normalize_embeddings=True
                        )

                        # 对该用户的所有推文向量取平均
                        user_embedding = tweet_embeddings.mean(dim=0)
                        tweets_list.append(user_embedding.cpu())

            # 堆叠成张量 [num_users, 1024]
            tweets_tensor = torch.stack(tweets_list, 0)
            print(f'BGE-M3 特征提取完成. Shape: {tweets_tensor.shape}')

            # 使用 MLP 降维
            if use_mlp:
                print(f'使用 MLP 降维到 {output_dim} 维...')

                mlp = MLP(input_size=1024, output_size=output_dim, hidden_size=hidden_dim).to(device)
                mlp.eval()

                with torch.no_grad():
                    tweets_tensor = mlp(tweets_tensor.to(device)).cpu()

                print(f'降维完成. Shape: {tweets_tensor.shape}')


            tweets_tensor = tweets_tensor.to(self.device)

            if self.save:
                os.makedirs('./pt_data', exist_ok=True)
                torch.save(tweets_tensor, path)
                print(f'特征已保存至 {path}')
        else:
            print(f'从 {path} 加载预计算的推文嵌入')
            tweets_tensor = torch.load(path).to(self.device)

        print(f'tweets_embedding 完成. Shape: {tweets_tensor.shape}')
        return tweets_tensor

    metadata - num类
    def num_preprocess(self):
        print('加载数值型属性信息...', end=' ')
        path = os.path.join('./pt_data', 'num_prop.pt')
        if not os.path.exists(path):
            path = self.root
            # 粉丝数
            if not os.path.exits(path + "followers_count.pt"):
                followers_count = []
                for i in range (self.df_data_labeled.shape[0]):
                    if self.df_data_labeled['profile'][i] is None or self.df_data['profile'][i]['followers_count'] is None:
                        followers_count.append(0)
                    else:
                        followers_count.append(self.df_data_labeled['profile'][i]['followers_count'])
                followers_count=torch.tensor(np.array(followers_count,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(followers_count,path+"followers_count.pt")

            # 关注数
            friend_count = []
            for i in range(self.df_data.shape[0]):
                if self.df_data['profile'][i] is None or self.df_data['profile'][i]['friends_count'] is None:
                    friends_count.append(0)
                else:
                    friends_count.append(self.df_data['profile'][i]['friends_count'])
            friends_count = torch.tensor(np.array(friends_count, dtype=np.float32)).to(self.device)
            if self.save:
                torch.save(friends_count, path + 'friends_count.pt')



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


def create_dataloader(root='./processed_data', device='cuda', process=True, save=True,
                      use_mlp=True, output_dim=64, hidden_dim=512):
    print('开始数据预处理...')

    # 初始化数据集
    print('\n[1/8] 初始化 Twibot20 数据集...')
    dataset = Twibot20(root=root, device=device, process=process, save=save)


    # 生成简介嵌入特征
    print('\n[4/8] 生成简介嵌入特征...')
    des_features = dataset.Des_embbeding(use_mlp=use_mlp, output_dim=output_dim, hidden_dim=hidden_dim)
    print(f'✓ 简介嵌入特征生成完成，形状: {des_features.shape}')

    # 生成推文嵌入特征
    print('\n[6/8] 生成推文嵌入特征...')
    tweets_features = dataset.tweets_embedding(use_mlp=use_mlp, output_dim=output_dim, hidden_dim=hidden_dim)
    print(f'✓ 推文嵌入特征生成完成，形状: {tweets_features.shape}')


    # 保存数据集信息摘要
    if save:
        print('\n保存数据集信息摘要...')
        summary = {
            'des_feature_dim': des_features.shape[1] if len(des_features.shape) > 1 else des_features.shape[0],
            'tweets_feature_dim': tweets_features.shape[1] if len(tweets_features.shape) > 1 else tweets_features.shape[0],
            'use_mlp': use_mlp,
            'output_dim': output_dim if use_mlp else 1024,
            'device': device,
            'created_at': dt.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        summary_path = os.path.join(root, 'dataset_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f'✓ 数据集摘要已保存至: {summary_path}')


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



