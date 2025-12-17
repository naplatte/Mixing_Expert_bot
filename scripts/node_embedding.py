import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime as dt
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoModel, AutoTokenizer
from torch import nn


class NodeEmbeddingGenerator:
    """
    生成图节点的初始特征表示
    节点特征由des、数值、类别、post四个模态的encoder输出拼接而成
    每个模态输出32维，总共128维
    """
    
    def __init__(self, device='cuda', batch_size=64):
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.data_root = '../../autodl-fs'
        self.save_dir = '../../autodl-fs/node_embedding'
        
        print(f'使用设备: {self.device}')
        
        self.df_data = None
        self.num_nodes = 0
        
    def load_data(self):
        """加载所有数据（包括支持集）"""
        print('='*60)
        print('加载数据...')
        print('='*60)
        
        df_train = pd.read_json(os.path.join(self.data_root, 'Data/train.json'))
        df_test = pd.read_json(os.path.join(self.data_root, 'Data/test.json'))
        df_dev = pd.read_json(os.path.join(self.data_root, 'Data/dev.json'))
        df_support = pd.read_json(os.path.join(self.data_root, 'Data/support.json'))
        
        df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
        df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]
        df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]
        df_support = df_support.iloc[:, [0, 1, 2, 3]]
        df_support['label'] = 'None'
        
        self.df_data = pd.concat([df_train, df_dev, df_test, df_support], ignore_index=True)
        self.num_nodes = self.df_data.shape[0]
        
        print(f'总节点数: {self.num_nodes}')
        print(f'  - 带标签节点: {df_train.shape[0] + df_dev.shape[0] + df_test.shape[0]}')
        print(f'  - 支持集节点: {df_support.shape[0]}')
        print('数据加载完成\n')
        
    def preprocess_description(self):
        """预处理所有节点的简介数据"""
        print('预处理简介数据...')
        path = os.path.join(self.save_dir, 'description_all.npy')
        
        if os.path.exists(path):
            print('  从缓存加载...')
            description = np.load(path, allow_pickle=True)
        else:
            description = []
            for i in tqdm(range(self.num_nodes), desc='  处理简介'):
                if self.df_data['profile'][i] is None or self.df_data['profile'][i]['description'] is None:
                    description.append('None')
                else:
                    description.append(self.df_data['profile'][i]['description'])
            description = np.array(description)
            
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(path, description)
        
        print(f'  简介数据: {len(description)} 条\n')
        return description
    
    def preprocess_post(self):
        """预处理所有节点的推文数据（参考dataset.py的tweets_preprogress）"""
        print('预处理推文数据...')
        path = os.path.join(self.save_dir, 'post_all.npy')
        
        if os.path.exists(path):
            print('  从缓存加载...')
            posts = np.load(path, allow_pickle=True)
            try:
                posts = posts.tolist()
            except Exception:
                pass
        else:
            posts = []
            for i in tqdm(range(self.num_nodes), desc='  处理推文'):
                one_user_tweets = []
                if self.df_data['tweet'][i] is None:
                    one_user_tweets.append('')
                else:
                    for each in self.df_data['tweet'][i]:
                        one_user_tweets.append(each)
                posts.append(one_user_tweets)
            
            os.makedirs(self.save_dir, exist_ok=True)
            posts_obj = np.array(posts, dtype=object)
            np.save(path, posts_obj, allow_pickle=True)
        
        print(f'  推文数据: {len(posts)} 条\n')
        return posts
    
    def preprocess_numerical(self):
        """预处理所有节点的数值类数据"""
        print('预处理数值类数据...')
        path = os.path.join(self.save_dir, 'num_properties_all_tensor.pt')
        
        if os.path.exists(path):
            print('  从缓存加载...')
            num_prop = torch.load(path).to(self.device)
        else:
            followers_count = []
            friends_count = []
            screen_name_length = []
            favourites_count = []
            active_days = []
            statuses_count = []
            
            date0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
            
            for i in tqdm(range(self.num_nodes), desc='  提取数值特征'):
                if self.df_data['profile'][i] is None:
                    followers_count.append(0)
                    friends_count.append(0)
                    screen_name_length.append(0)
                    favourites_count.append(0)
                    active_days.append(0)
                    statuses_count.append(0)
                else:
                    profile = self.df_data['profile'][i]
                    
                    followers_count.append(profile['followers_count'] if profile['followers_count'] is not None else 0)
                    friends_count.append(profile['friends_count'] if profile['friends_count'] is not None else 0)
                    screen_name_length.append(len(profile['screen_name']) if profile['screen_name'] is not None else 0)
                    favourites_count.append(profile['favourites_count'] if profile['favourites_count'] is not None else 0)
                    statuses_count.append(int(profile['statuses_count']) if profile['statuses_count'] is not None else 0)
                    
                    if profile['created_at'] is not None:
                        date = dt.strptime(profile['created_at'], '%a %b %d %X %z %Y ')
                        active_days.append((date0 - date).days)
                    else:
                        active_days.append(0)
            
            followers_count = torch.tensor(np.array(followers_count, dtype=np.float32))
            friends_count = torch.tensor(np.array(friends_count, dtype=np.float32))
            screen_name_length = torch.tensor(np.array(screen_name_length, dtype=np.float32))
            favourites_count = torch.tensor(np.array(favourites_count, dtype=np.float32))
            active_days = torch.tensor(np.array(active_days, dtype=np.float32))
            statuses_count = torch.tensor(np.array(statuses_count, dtype=np.float32))
            
            print('  Z-score标准化...')
            followers_count = (followers_count - followers_count.mean()) / followers_count.std()
            friends_count = (friends_count - friends_count.mean()) / friends_count.std()
            screen_name_length = (screen_name_length - screen_name_length.mean()) / screen_name_length.std()
            favourites_count = (favourites_count - favourites_count.mean()) / favourites_count.std()
            active_days = (active_days - active_days.mean()) / active_days.std()
            statuses_count = (statuses_count - statuses_count.mean()) / statuses_count.std()
            
            num_prop = torch.cat((
                followers_count.reshape([self.num_nodes, 1]),
                friends_count.reshape([self.num_nodes, 1]),
                favourites_count.reshape([self.num_nodes, 1]),
                statuses_count.reshape([self.num_nodes, 1]),
                screen_name_length.reshape([self.num_nodes, 1]),
                active_days.reshape([self.num_nodes, 1])
            ), 1).to(self.device)
            
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(num_prop, path)
        
        print(f'  数值特征: {num_prop.shape}\n')
        return num_prop
    
    def preprocess_categorical(self):
        """预处理所有节点的类别类数据"""
        print('预处理类别类数据...')
        path = os.path.join(self.save_dir, 'cat_properties_all_tensor.pt')
        
        if os.path.exists(path):
            print('  从缓存加载...')
            cat_prop = torch.load(path).to(self.device)
        else:
            category_properties = []
            properties = ['protected', 'geo_enabled', 'verified', 'contributors_enabled', 'is_translator',
                         'is_translation_enabled', 'profile_background_tile', 'profile_use_background_image',
                         'has_extended_profile', 'default_profile', 'default_profile_image']
            
            for i in tqdm(range(self.num_nodes), desc='  提取类别特征'):
                prop = []
                if self.df_data['profile'][i] is None:
                    for j in range(11):
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
            
            cat_prop = torch.tensor(np.array(category_properties, dtype=np.float32)).to(self.device)
            
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(cat_prop, path)
        
        print(f'  类别特征: {cat_prop.shape}\n')
        return cat_prop
    
    def encode_description(self, descriptions):
        """
        使用RoBERTa encoder编码简介数据
        768维 -> 32维
        """
        print('='*60)
        print('编码简介数据 (RoBERTa)')
        print('='*60)
        
        save_path = os.path.join(self.save_dir, 'node_des_embeddings.pt')
        
        if os.path.exists(save_path):
            print('从缓存加载简介编码...')
            des_embeddings = torch.load(save_path).to(self.device)
            print(f'简介编码维度: {des_embeddings.shape}\n')
            return des_embeddings
        
        model_name = 'roberta-base'
        print(f'加载模型: {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone_model = AutoModel.from_pretrained(model_name)
        backbone_model.eval()
        backbone_model = backbone_model.to(self.device)
        
        for param in backbone_model.parameters():
            param.requires_grad = False
        
        projection = nn.Linear(768, 32).to(self.device)
        
        all_embeddings = []
        
        num_batches = (self.num_nodes + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc='编码简介'):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_nodes)
                batch_descriptions = descriptions[start_idx:end_idx]
                
                cleaned_descriptions = []
                for desc in batch_descriptions:
                    desc_str = str(desc).strip()
                    if desc_str == '' or desc_str.lower() == 'none':
                        desc_str = ''
                    cleaned_descriptions.append(desc_str)
                
                encoded = tokenizer(
                    cleaned_descriptions,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = backbone_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * attention_mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = attention_mask_expanded.sum(dim=1)
                sentence_vectors = sum_hidden / sum_mask.clamp(min=1)
                
                projected = projection(sentence_vectors)
                all_embeddings.append(projected.cpu())
        
        des_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(des_embeddings, save_path)
        
        print(f'简介编码完成: {des_embeddings.shape}\n')
        
        del backbone_model, tokenizer, projection
        torch.cuda.empty_cache()
        
        return des_embeddings
    
    def encode_post(self, posts):
        """
        使用RoBERTa encoder编码推文数据
        768维 -> 32维
        """
        print('='*60)
        print('编码推文数据 (RoBERTa)')
        print('='*60)
        
        save_path = os.path.join(self.save_dir, 'node_post_embeddings.pt')
        
        if os.path.exists(save_path):
            print('从缓存加载推文编码...')
            post_embeddings = torch.load(save_path).to(self.device)
            print(f'推文编码维度: {post_embeddings.shape}\n')
            return post_embeddings
        
        model_name = 'roberta-base'
        print(f'加载模型: {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone_model = AutoModel.from_pretrained(model_name)
        backbone_model.eval()
        backbone_model = backbone_model.to(self.device)
        
        for param in backbone_model.parameters():
            param.requires_grad = False
        
        projection = nn.Linear(768, 32).to(self.device)
        
        all_embeddings = []
        
        num_batches = (self.num_nodes + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc='编码推文'):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_nodes)
                batch_posts = posts[start_idx:end_idx]
                
                cleaned_posts = []
                for user_tweets in batch_posts:
                    if user_tweets is None or len(user_tweets) == 0:
                        cleaned_posts.append('')
                    else:
                        tweet_texts = [str(t).strip() for t in user_tweets[:20] if t is not None and str(t).strip() != '']
                        combined_text = ' '.join(tweet_texts) if tweet_texts else ''
                        cleaned_posts.append(combined_text)
                
                encoded = tokenizer(
                    cleaned_posts,
                    max_length=256,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = backbone_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * attention_mask_expanded
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = attention_mask_expanded.sum(dim=1)
                sentence_vectors = sum_hidden / sum_mask.clamp(min=1)
                
                projected = projection(sentence_vectors)
                all_embeddings.append(projected.cpu())
        
        post_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(post_embeddings, save_path)
        
        print(f'推文编码完成: {post_embeddings.shape}\n')
        
        del backbone_model, tokenizer, projection
        torch.cuda.empty_cache()
        
        return post_embeddings
    
    def encode_numerical(self, num_features):
        """
        使用线性层编码数值数据
        6维 -> 32维
        """
        print('='*60)
        print('编码数值数据 (Linear Encoder)')
        print('='*60)
        
        save_path = os.path.join(self.save_dir, 'node_num_embeddings.pt')
        
        if os.path.exists(save_path):
            print('从缓存加载数值编码...')
            num_embeddings = torch.load(save_path).to(self.device)
            print(f'数值编码维度: {num_embeddings.shape}\n')
            return num_embeddings
        
        encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        ).to(self.device)
        
        with torch.no_grad():
            num_embeddings = encoder(num_features)
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(num_embeddings, save_path)
        
        print(f'数值编码完成: {num_embeddings.shape}\n')
        
        del encoder
        torch.cuda.empty_cache()
        
        return num_embeddings
    
    def encode_categorical(self, cat_features):
        """
        使用线性层编码类别数据
        11维 -> 32维
        """
        print('='*60)
        print('编码类别数据 (Linear Encoder)')
        print('='*60)
        
        save_path = os.path.join(self.save_dir, 'node_cat_embeddings.pt')
        
        if os.path.exists(save_path):
            print('从缓存加载类别编码...')
            cat_embeddings = torch.load(save_path).to(self.device)
            print(f'类别编码维度: {cat_embeddings.shape}\n')
            return cat_embeddings
        
        encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        ).to(self.device)
        
        with torch.no_grad():
            cat_embeddings = encoder(cat_features)
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(cat_embeddings, save_path)
        
        print(f'类别编码完成: {cat_embeddings.shape}\n')
        
        del encoder
        torch.cuda.empty_cache()
        
        return cat_embeddings
    
    def generate_node_embeddings(self):
        """生成最终的节点初始特征（128维）"""
        print('='*60)
        print('生成节点初始特征')
        print('='*60)
        
        final_path = os.path.join(self.save_dir, 'node_initial_features.pt')
        
        if os.path.exists(final_path):
            print('节点初始特征已存在，跳过生成')
            node_features = torch.load(final_path).to(self.device)
            print(f'节点特征维度: {node_features.shape}')
            return node_features
        
        self.load_data()
        
        descriptions = self.preprocess_description()
        posts = self.preprocess_post()
        num_features = self.preprocess_numerical()
        cat_features = self.preprocess_categorical()
        
        des_embeddings = self.encode_description(descriptions)
        post_embeddings = self.encode_post(posts)
        num_embeddings = self.encode_numerical(num_features)
        cat_embeddings = self.encode_categorical(cat_features)
        
        print('拼接四个模态的特征...')
        node_features = torch.cat([des_embeddings, post_embeddings, num_embeddings, cat_embeddings], dim=1)
        
        print(f'  - 简介编码: {des_embeddings.shape[1]}维')
        print(f'  - 推文编码: {post_embeddings.shape[1]}维')
        print(f'  - 数值编码: {num_embeddings.shape[1]}维')
        print(f'  - 类别编码: {cat_embeddings.shape[1]}维')
        print(f'  - 总维度: {node_features.shape[1]}维')
        
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(node_features, final_path)
        
        print(f'\n节点初始特征已保存: {final_path}')
        print(f'特征维度: {node_features.shape}')
        print('='*60)
        
        return node_features


def main():
    """主函数"""
    print('\n' + '='*60)
    print('节点嵌入生成器')
    print('='*60 + '\n')
    
    generator = NodeEmbeddingGenerator(device='cuda', batch_size=64)
    
    node_features = generator.generate_node_embeddings()
    
    print('\n✓ 所有节点特征生成完成！')
    print(f'✓ 总节点数: {node_features.shape[0]}')
    print(f'✓ 特征维度: {node_features.shape[1]}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
