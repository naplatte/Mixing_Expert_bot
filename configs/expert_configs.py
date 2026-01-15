"""专家配置模块 - 支持Cat/Num/Des/Post/Graph专家的训练与全节点嵌入生成"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import Twibot20
from src.model import DesExpertMoE, PostExpert, GraphExpert, CatExpert, NumExpert

PROJECT_ROOT = Path(__file__).parent.parent


class SimpleDataset(Dataset):
    def __init__(self, features, labels, mode='train', filter_fn=None, feature_key='features'):
        self.mode = mode
        self.feature_key = feature_key
        if filter_fn:
            valid_idx = [i for i, f in enumerate(features) if filter_fn(f)]
            self.features = [features[i] for i in valid_idx]
            self.labels = [labels[i] for i in valid_idx]
            print(f"  [{mode}集] 样本: {len(self.features)}/{len(features)} (过滤{len(features)-len(self.features)}个)")
        else:
            self.features = features if isinstance(features, list) else features
            self.labels = labels
            print(f"  [{mode}集] 样本: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        if not isinstance(feat, torch.Tensor):
            if isinstance(feat, (list, str)):
                return {self.feature_key: feat, 'label': torch.tensor(label, dtype=torch.float32)}
            feat = torch.tensor(feat, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        return {self.feature_key: feat, 'label': label}


class GraphDataset(Dataset):
    def __init__(self, node_indices, labels):
        self.node_indices = node_indices
        self.labels = labels

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        return {
            'node_index': torch.tensor(self.node_indices[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def collate_posts_fn(batch):
    return {
        'posts_text_list': [item['features'] for item in batch],
        'label': torch.stack([item['label'] for item in batch])
    }


def collate_graph_fn(batch):
    return {
        'node_indices': torch.stack([item['node_index'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def _load_dataset(dataset_path, device, twibot_dataset=None):
    if twibot_dataset is None:
        print("加载数据...")
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    return twibot_dataset


def _get_splits(twibot_dataset):
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    return list(train_idx), list(val_idx), list(test_idx)


def _create_loaders(train_data, val_data, test_data, train_labels, val_labels, test_labels,
                    batch_size, dataset_class=SimpleDataset, filter_fn=None,
                    collate_fn=None, feature_key='features'):
    train_ds = dataset_class(train_data, train_labels, 'train', filter_fn, feature_key) if dataset_class == SimpleDataset \
               else dataset_class(train_data, train_labels)
    val_ds = dataset_class(val_data, val_labels, 'val', filter_fn, feature_key) if dataset_class == SimpleDataset \
             else dataset_class(val_data, val_labels)
    test_ds = dataset_class(test_data, test_labels, 'test', filter_fn, feature_key) if dataset_class == SimpleDataset \
              else dataset_class(test_data, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def _build_config(name, model, train_loader, val_loader, test_loader,
                  extract_fn, device, checkpoint_dir, learning_rate, weight_decay=0.01,
                  max_grad_norm=1.0, early_stopping_patience=5, **extra):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=learning_rate, weight_decay=weight_decay)
    return {
        'name': name,
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': nn.BCELoss(),
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn,
        'max_grad_norm': max_grad_norm,
        'early_stopping_patience': early_stopping_patience,
        **extra
    }


def create_des_expert_config(dataset_path=None, batch_size=32, learning_rate=5e-4,
                              device='cuda', checkpoint_dir='../../autodl-fs/model',
                              dropout=0.3, twibot_dataset=None, **kwargs):
    dataset_path = dataset_path or str(PROJECT_ROOT / 'processed_data')
    print(f"\n{'='*60}\n配置 Description Expert\n{'='*60}")
    twibot = _load_dataset(dataset_path, device, twibot_dataset)
    descriptions = twibot.Des_preprocess()
    labels = twibot.load_labels().cpu().numpy()
    if isinstance(descriptions, np.ndarray):
        descriptions = descriptions.tolist()
    train_idx, val_idx, test_idx = _get_splits(twibot)

    def is_valid(desc):
        s = str(desc).strip()
        return s and s.lower() != 'none'

    train_loader, val_loader, test_loader = _create_loaders(
        [descriptions[i] for i in train_idx], [descriptions[i] for i in val_idx],
        [descriptions[i] for i in test_idx],
        labels[train_idx], labels[val_idx], labels[test_idx],
        batch_size, filter_fn=is_valid, feature_key='description_text'
    )
    model = DesExpertMoE(device=device, dropout=dropout).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def extract_fn(batch, device):
        return (batch['description_text'],), batch['label'].to(device).unsqueeze(1)

    return _build_config('des', model, train_loader, val_loader, test_loader,
                        extract_fn, device, checkpoint_dir, learning_rate)


def create_post_expert_config(dataset_path=None, batch_size=32, learning_rate=5e-4,
                               device='cuda', checkpoint_dir='../../autodl-fs/model',
                               dropout=0.3, twibot_dataset=None, **kwargs):
    dataset_path = dataset_path or str(PROJECT_ROOT / 'processed_data')
    print(f"\n{'='*60}\n配置 Post Expert\n{'='*60}")
    twibot = _load_dataset(dataset_path, device, twibot_dataset)
    posts = twibot.tweets_preprogress()
    labels = twibot.load_labels().cpu().numpy()
    if isinstance(posts, np.ndarray):
        posts = posts.tolist()
    train_idx, val_idx, test_idx = _get_splits(twibot)

    def is_valid(user_posts):
        if isinstance(user_posts, list):
            cleaned = [str(p).strip() for p in user_posts if str(p).strip() and str(p).lower() != 'none']
            return len(cleaned) > 0
        return False

    train_loader, val_loader, test_loader = _create_loaders(
        [posts[i] for i in train_idx], [posts[i] for i in val_idx], [posts[i] for i in test_idx],
        labels[train_idx], labels[val_idx], labels[test_idx],
        batch_size, filter_fn=is_valid, collate_fn=collate_posts_fn
    )
    model = PostExpert(device=device, dropout=dropout).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def extract_fn(batch, device):
        return (batch['posts_text_list'],), batch['label'].to(device).unsqueeze(1)

    return _build_config('post', model, train_loader, val_loader, test_loader,
                        extract_fn, device, checkpoint_dir, learning_rate)


def create_cat_expert_config(dataset_path=None, batch_size=64, learning_rate=1e-3,
                              device='cuda', checkpoint_dir='../../autodl-fs/model',
                              dropout=0.2, twibot_dataset=None, **kwargs):
    """类别属性专家 - 3个属性: 是否私密、是否认证、是否默认头像"""
    dataset_path = dataset_path or str(PROJECT_ROOT / 'processed_data')
    print(f"\n{'='*60}\n配置 Category Expert (3个属性)\n{'='*60}")
    twibot = _load_dataset(dataset_path, device, twibot_dataset)
    cat_features = twibot.cat_prop_preprocess()
    labels = twibot.load_labels().cpu().numpy()
    if isinstance(cat_features, torch.Tensor):
        cat_features = cat_features.cpu()
    input_dim = cat_features.shape[1]
    print(f"  类别属性维度: {input_dim}")
    train_idx, val_idx, test_idx = _get_splits(twibot)
    train_loader, val_loader, test_loader = _create_loaders(
        cat_features[train_idx], cat_features[val_idx], cat_features[test_idx],
        labels[train_idx], labels[val_idx], labels[test_idx],
        batch_size, feature_key='cat_features'
    )
    model = CatExpert(input_dim=input_dim, dropout=dropout, device=device).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    def extract_fn(batch, device):
        return (batch['cat_features'].to(device),), batch['label'].to(device).unsqueeze(1)

    return _build_config('cat', model, train_loader, val_loader, test_loader,
                        extract_fn, device, checkpoint_dir, learning_rate)


def create_num_expert_config(dataset_path=None, batch_size=64, learning_rate=1e-3,
                              device='cuda', checkpoint_dir='../../autodl-fs/model',
                              dropout=0.2, twibot_dataset=None, **kwargs):
    """数值属性专家 - 5个属性: followers/friends/statuses_count, screen_name_length, active_days"""
    dataset_path = dataset_path or str(PROJECT_ROOT / 'processed_data')
    print(f"\n{'='*60}\n配置 Numerical Expert (5个属性)\n{'='*60}")
    twibot = _load_dataset(dataset_path, device, twibot_dataset)
    num_features = twibot.num_prop_preprocess()
    labels = twibot.load_labels().cpu().numpy()
    if isinstance(num_features, torch.Tensor):
        num_features = num_features.cpu()
    input_dim = num_features.shape[1]
    print(f"  数值属性维度: {input_dim}")
    train_idx, val_idx, test_idx = _get_splits(twibot)
    train_loader, val_loader, test_loader = _create_loaders(
        num_features[train_idx], num_features[val_idx], num_features[test_idx],
        labels[train_idx], labels[val_idx], labels[test_idx],
        batch_size, feature_key='num_features'
    )
    model = NumExpert(input_dim=input_dim, dropout=dropout, device=device).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    def extract_fn(batch, device):
        return (batch['num_features'].to(device),), batch['label'].to(device).unsqueeze(1)

    return _build_config('num', model, train_loader, val_loader, test_loader,
                        extract_fn, device, checkpoint_dir, learning_rate)


def create_graph_expert_config(dataset_path=None, batch_size=128, learning_rate=1e-3,
                                device='cuda', checkpoint_dir='../../autodl-fs/model',
                                embedding_dim=128, dropout=0.3, twibot_dataset=None, **kwargs):
    dataset_path = dataset_path or str(PROJECT_ROOT / 'processed_data')
    print(f"\n{'='*60}\n配置 Graph Expert\n{'='*60}")
    twibot = _load_dataset(dataset_path, device, twibot_dataset)
    labels = twibot.load_labels().cpu().numpy()
    edge_index, edge_type = twibot.build_graph()
    print(f"  边数: {edge_index.shape[1]}")
    train_idx, val_idx, test_idx = _get_splits(twibot)
    num_all_nodes = len(twibot.df_data)
    print(f"  总节点数: {num_all_nodes}")
    train_ds = GraphDataset(train_idx, labels[train_idx])
    val_ds = GraphDataset(val_idx, labels[val_idx])
    test_ds = GraphDataset(test_idx, labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_fn)
    model = GraphExpert(num_nodes=num_all_nodes, embedding_dim=embedding_dim,
                        dropout=dropout, device=device).to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def extract_fn(batch, device):
        node_indices = batch['node_indices'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        return (node_indices, edge_index, edge_type), labels

    return _build_config('graph', model, train_loader, val_loader, test_loader,
                        extract_fn, device, checkpoint_dir, learning_rate,
                        edge_index=edge_index, edge_type=edge_type)


EXPERT_CONFIGS = {
    'des': create_des_expert_config,
    'post': create_post_expert_config,
    'cat': create_cat_expert_config,
    'num': create_num_expert_config,
    'graph': create_graph_expert_config,
}


def get_expert_config(expert_name, **kwargs):
    if expert_name not in EXPERT_CONFIGS:
        raise ValueError(f"未知专家: {expert_name}. 可用: {list(EXPERT_CONFIGS.keys())}")
    return EXPERT_CONFIGS[expert_name](**kwargs)
