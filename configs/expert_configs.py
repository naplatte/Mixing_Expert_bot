"""
专家配置模块
定义各个专家的配置函数，包括数据集、模型、优化器等
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import Twibot20
from src.model import DesExpert, TweetsExpert

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


# ==================== Description Expert ====================

class DescriptionDataset(Dataset):
    """Description 专家数据集"""
    def __init__(self, descriptions, labels, tokenizer, max_length=128, mode='train'):
        """
        Args:
            descriptions: 简介列表
            labels: 标签列表
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            mode: 'train' | 'val' | 'test'
                  所有阶段都过滤掉空简介的样本
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # 所有阶段都过滤掉空简介的样本
        valid_indices = []
        for idx, desc in enumerate(descriptions):
            desc_str = str(desc).strip()
            if desc_str != '' and desc_str.lower() != 'none':
                valid_indices.append(idx)

        self.descriptions = [descriptions[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]

        filtered_count = len(descriptions) - len(self.descriptions)
        print(f"  [{mode}集] 有效样本: {len(self.descriptions)}/{len(descriptions)} (过滤 {filtered_count} 个空简介样本)")

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        label = self.labels[idx]

        # Tokenize（保证非空）
        encoded = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def create_des_expert_config(
    dataset_path=None,
    batch_size=32,
    learning_rate=2e-5,
    device='cuda',
    checkpoint_dir='../../autodl-tmp/checkpoints',
    bert_model_name='bert-base-uncased',
    freeze_bert=True
):
    """
    创建 Description Expert 配置

    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / 'processed_data')

    print(f"\n{'='*60}")
    print(f"配置 Description Expert")
    print(f"{'='*60}")

    # 加载数据
    print("加载数据...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)

    descriptions = twibot_dataset.Des_preprocess()
    labels = twibot_dataset.load_labels()

    if isinstance(descriptions, np.ndarray):
        descriptions = descriptions.tolist()
    labels = labels.cpu().numpy()

    # 获取训练/验证/测试集索引
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    # 划分数据集
    train_descriptions = [descriptions[i] for i in train_idx]
    train_labels = labels[train_idx]

    val_descriptions = [descriptions[i] for i in val_idx]
    val_labels = labels[val_idx]

    test_descriptions = [descriptions[i] for i in test_idx]
    test_labels = labels[test_idx]

    print(f"  训练集: {len(train_descriptions)} 样本")
    print(f"  验证集: {len(val_descriptions)} 样本")
    print(f"  测试集: {len(test_descriptions)} 样本")

    # 创建 tokenizer
    print(f"初始化 tokenizer ({bert_model_name})...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = DescriptionDataset(train_descriptions, train_labels, tokenizer, mode='train')
    val_dataset = DescriptionDataset(val_descriptions, val_labels, tokenizer, mode='val')
    test_dataset = DescriptionDataset(test_descriptions, test_labels, tokenizer, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    print("初始化模型...")
    model = DesExpert(bert_model_name=bert_model_name, freeze_bert=freeze_bert).to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器和损失函数
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.BCELoss()

    # 数据提取函数
    def extract_fn(batch, device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        return (input_ids, attention_mask), labels

    return {
        'name': 'des',
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn
    }


# ==================== Tweets Expert ====================

class TweetsDataset(Dataset):
    def __init__(self, tweets_list, labels, mode='train'):
        """
        Args:
            tweets_list: 推文列表
            labels: 标签列表
            mode: 'train' | 'val' | 'test'
                  所有阶段都过滤掉没有推文的样本
        """
        self.mode = mode

        # 所有阶段都过滤掉没有推文的样本
        valid_indices = []
        for idx, user_tweets in enumerate(tweets_list):
            cleaned = self._clean_tweets(user_tweets)
            if len(cleaned) > 0:
                valid_indices.append(idx)

        self.tweets_list = [tweets_list[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]

        filtered_count = len(tweets_list) - len(self.tweets_list)
        print(f"  [{mode}集] 有效样本: {len(self.tweets_list)}/{len(tweets_list)} (过滤 {filtered_count} 个无推文样本)")

    def _clean_tweets(self, user_tweets):
        """清理推文文本"""
        cleaned = []
        if isinstance(user_tweets, list):
            for tweet in user_tweets:
                tweet_str = str(tweet).strip()
                if tweet_str != '' and tweet_str != 'None':
                    cleaned.append(tweet_str)
        return cleaned

    def __len__(self):
        return len(self.tweets_list)

    def __getitem__(self, idx):
        user_tweets = self.tweets_list[idx]
        label = self.labels[idx]

        # 清理推文文本（保证非空）
        cleaned_tweets = self._clean_tweets(user_tweets)

        return {
            'tweets_text': cleaned_tweets,
            'label': torch.tensor(label, dtype=torch.float32)
        }

# 将一个 batch 中每个样本的推文文本列表和标签整理成字典供模型输入
def collate_tweets_fn(batch):
    tweets_text_lists = [item['tweets_text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    return {
        'tweets_text_list': tweets_text_lists,
        'label': labels
    }

def create_tweets_expert_config(
    dataset_path=None,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda',
    checkpoint_dir='../../autodl-tmp/checkpoints',
    roberta_model_name='distilroberta-base'
):
    """
    创建 Tweets Expert 配置
    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / 'processed_data')

    print(f"\n{'='*60}")
    print(f"配置 Tweets Expert")
    print(f"{'='*60}")

    # 加载数据
    print("加载数据...")
    twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)

    tweets_list = twibot_dataset.tweets_preprogress()
    labels = twibot_dataset.load_labels()

    if isinstance(tweets_list, np.ndarray):
        tweets_list = tweets_list.tolist()
    labels = labels.cpu().numpy()

    # 获取训练/验证/测试集索引
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    # 划分数据集
    train_tweets = [tweets_list[i] for i in train_idx]
    train_labels = labels[train_idx]

    val_tweets = [tweets_list[i] for i in val_idx]
    val_labels = labels[val_idx]

    test_tweets = [tweets_list[i] for i in test_idx]
    test_labels = labels[test_idx]

    print(f"  训练集: {len(train_tweets)} 样本")
    print(f"  验证集: {len(val_tweets)} 样本")
    print(f"  测试集: {len(test_tweets)} 样本")

    # 统计推文数量
    train_tweet_counts = [len(tweets) if isinstance(tweets, list) else 0 for tweets in train_tweets]
    print(f"  训练集平均推文数: {np.mean(train_tweet_counts):.2f}")

    # 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = TweetsDataset(train_tweets, train_labels, mode='train')
    val_dataset = TweetsDataset(val_tweets, val_labels, mode='val')
    test_dataset = TweetsDataset(test_tweets, test_labels, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tweets_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tweets_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tweets_fn)

    # 初始化模型
    print(f"初始化模型 ({roberta_model_name})...")
    model = TweetsExpert(roberta_model_name=roberta_model_name, device=device).to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # 数据提取函数
    def extract_fn(batch, device):
        tweets_text_list = batch['tweets_text_list']
        labels = batch['label'].to(device).unsqueeze(1)
        return (tweets_text_list,), labels

    return {
        'name': 'tweets',
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn
    }


# ==================== 配置注册表 ====================

EXPERT_CONFIGS = {
    'des': create_des_expert_config,
    'tweets': create_tweets_expert_config,
    # 后续可以添加更多专家配置
    # 'graph': create_graph_expert_config,
    # 'metadata': create_metadata_expert_config,
}

# 获取专家配置
def get_expert_config(expert_name, **kwargs):
    """
    Args:
        expert_name: 专家名称 ('des', 'tweets', 'graph', etc.)
        **kwargs: 传递给配置函数的参数

    Returns:
        dict: 专家配置字典
    """
    if expert_name not in EXPERT_CONFIGS:
        raise ValueError(f"未知的专家名称: {expert_name}. 可用选项: {list(EXPERT_CONFIGS.keys())}")

    config_fn = EXPERT_CONFIGS[expert_name]

    # 过滤kwargs，只传递专家配置函数接受的参数
    import inspect
    sig = inspect.signature(config_fn)
    valid_params = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return config_fn(**filtered_kwargs)

