"""
专家配置模块
定义各个专家的配置函数，包括数据集、模型、优化器等
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import Twibot20
from src.model import DesExpert, DesExpertMoE, TweetsExpert, GraphExpert

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


# ==================== 辅助函数 ====================

def _extract_expert_features_with_mask(expert_name, node_indices, dataset_path, checkpoint_dir, device, twibot_dataset=None):
    """
    从已训练的专家模型中提取特征

    Args:
        expert_name: 专家名称 ('des', 'tweets')
        node_indices: 需要提取特征的节点索引列表
        dataset_path: 数据集路径
        checkpoint_dir: 模型检查点目录
        device: 设备
        twibot_dataset: 预加载的数据集（可选）

    Returns:
        embeddings: [num_nodes, expert_dim] 特征向量
        mask: [num_nodes] bool tensor，标记哪些节点有有效特征
    """
    from tqdm import tqdm

    # 加载数据集（如果未提供）
    if twibot_dataset is None:
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)

    # 加载已训练的专家模型
    checkpoint_path = Path(checkpoint_dir) / f'{expert_name}_expert_best.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到专家模型检查点: {checkpoint_path}")

    print(f"    加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 根据专家类型创建模型并加载权重
    if expert_name == 'des':
        model = DesExpert(device=device).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 获取描述数据
        descriptions = twibot_dataset.Des_preprocess()
        if isinstance(descriptions, np.ndarray):
            descriptions = descriptions.tolist()

        # 提取特征
        model.eval()
        embeddings_list = []
        valid_mask = []

        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(node_indices), batch_size), desc=f"  提取 {expert_name} 特征"):
                batch_indices = node_indices[i:i+batch_size]
                batch_descriptions = [descriptions[idx] for idx in batch_indices]

                # 检查是否有有效描述
                batch_valid = []
                for desc in batch_descriptions:
                    desc_str = str(desc).strip()
                    is_valid = desc_str != '' and desc_str.lower() != 'none'
                    batch_valid.append(is_valid)

                # 提取特征
                expert_repr, _ = model(batch_descriptions)
                embeddings_list.append(expert_repr.cpu())
                valid_mask.extend(batch_valid)

        embeddings = torch.cat(embeddings_list, dim=0)
        mask = torch.tensor(valid_mask, dtype=torch.bool)

    elif expert_name == 'tweets':
        model = TweetsExpert(device=device).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 获取推文数据
        tweets_list = twibot_dataset.tweets_preprogress()
        if isinstance(tweets_list, np.ndarray):
            tweets_list = tweets_list.tolist()

        # 提取特征
        model.eval()
        embeddings_list = []
        valid_mask = []

        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(node_indices), batch_size), desc=f"  提取 {expert_name} 特征"):
                batch_indices = node_indices[i:i+batch_size]
                batch_tweets = [tweets_list[idx] for idx in batch_indices]

                # 检查是否有有效推文
                batch_valid = []
                for user_tweets in batch_tweets:
                    if isinstance(user_tweets, list) and len(user_tweets) > 0:
                        cleaned = [str(t).strip() for t in user_tweets if str(t).strip() != '' and str(t).strip() != 'None']
                        is_valid = len(cleaned) > 0
                    else:
                        is_valid = False
                    batch_valid.append(is_valid)

                # 提取特征
                expert_repr, _ = model(batch_tweets)
                embeddings_list.append(expert_repr.cpu())
                valid_mask.extend(batch_valid)

        embeddings = torch.cat(embeddings_list, dim=0)
        mask = torch.tensor(valid_mask, dtype=torch.bool)

    else:
        raise ValueError(f"不支持的专家类型: {expert_name}")

    print(f"    特征形状: {embeddings.shape}, 有效样本: {mask.sum().item()}/{len(mask)}")

    return embeddings, mask


# ==================== Description Expert ====================

class DescriptionDataset(Dataset):
    """Description 专家数据集"""
    def __init__(self, descriptions, labels, mode='train'):
        """
        Args:
            descriptions: 简介列表
            labels: 标签列表
            mode: 'train' | 'val' | 'test'
                  所有阶段都过滤掉空简介的样本
        """
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

        return {
            'description_text': description,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def create_des_expert_config(
    dataset_path=None,
    batch_size=32,
    learning_rate=5e-4,
    weight_decay=0.01,
    device='cuda',
    checkpoint_dir='../../autodl-fs/model',
    model_name='microsoft/deberta-v3-base',
    max_grad_norm=1.0,
    dropout=0.3,
    early_stopping_patience=4,
    num_experts=4,  # MoE 中的专家数量（默认4个）
    top_k=2,        # Top-K 选择，每次只用权重最大的K个专家（默认2个）
    twibot_dataset=None
):
    """
    创建 Description Expert 配置 (使用 DeBERTa-v3-base + MoE + Top-K)

    Args:
        model_name: 预训练模型名称，默认为 'microsoft/deberta-v3-base'
        batch_size: 批次大小，默认32
        learning_rate: 学习率，默认5e-4 (0.0005)
        weight_decay: 权重衰减，默认0.01
        dropout: Dropout率，默认0.3
        max_grad_norm: 梯度裁剪阈值，默认1.0
        early_stopping_patience: 早停耐心值，默认4
        num_experts: MoE 中的专家数量，默认4
        top_k: 每次选择的专家数量（Top-K），默认2
        twibot_dataset: 预加载的Twibot20数据集对象（可选，避免重复加载）

    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / 'processed_data')

    print(f"\n{'='*60}")
    print(f"配置 Description Expert with MoE + Top-K (DeBERTa-v3-base)")
    print(f"{'='*60}")
    print(f"  模型: {model_name}")
    print(f"  专家数量: {num_experts}")
    print(f"  Top-K 选择: {top_k} (每次只用权重最大的{top_k}个专家)")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  Dropout: {dropout}")
    print(f"  梯度裁剪: {max_grad_norm}")
    print(f"  早停耐心值: {early_stopping_patience}")

    # 加载数据（如果没有预加载）
    if twibot_dataset is None:
        print("加载数据...")
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    else:
        print("使用预加载的数据集...")

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

    # 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = DescriptionDataset(train_descriptions, train_labels, mode='train')
    val_dataset = DescriptionDataset(val_descriptions, val_labels, mode='val')
    test_dataset = DescriptionDataset(test_descriptions, test_labels, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型 (使用 MoE + Top-K 版本)
    print("初始化 MoE + Top-K 模型...")
    model = DesExpertMoE(
        model_name=model_name,
        device=device,
        dropout=dropout,
        num_experts=num_experts,
        top_k=top_k,
        expert_dim=64,
        hidden_dim=768
    ).to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  - Gating Network + {num_experts} 个 MLP 专家 (Top-{top_k} 选择)")

    # 优化器和损失函数 - 使用权重衰减
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = nn.BCELoss()

    # 数据提取函数
    def extract_fn(batch, device):
        description_texts = batch['description_text']
        labels = batch['label'].to(device).unsqueeze(1)
        return (description_texts,), labels

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
        'extract_fn': extract_fn,
        'max_grad_norm': max_grad_norm,  # 梯度裁剪
        'early_stopping_patience': early_stopping_patience,  # 早停
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'num_experts': num_experts,  # MoE 专家数量
        'top_k': top_k  # Top-K 选择
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
    checkpoint_dir='../../autodl-fs/model',
    roberta_model_name='distilroberta-base',
    twibot_dataset=None
):
    """
    创建 Tweets Expert 配置

    Args:
        twibot_dataset: 预加载的Twibot20数据集对象（可选，避免重复加载）

    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / 'processed_data')

    print(f"\n{'='*60}")
    print(f"配置 Tweets Expert")
    print(f"{'='*60}")

    # 加载数据（如果没有预加载）
    if twibot_dataset is None:
        print("加载数据...")
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    else:
        print("使用预加载的数据集...")

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


# ==================== Graph Expert ====================

class GraphDataset(Dataset):
    """Graph Expert 数据集"""
    def __init__(self, node_indices, labels):
        """
        Args:
            node_indices: 节点索引列表
            labels: 标签列表
        """
        self.node_indices = node_indices
        self.labels = labels

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        return {
            'node_index': torch.tensor(self.node_indices[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def collate_graph_fn(batch):
    """图数据 collate 函数"""
    node_indices = torch.stack([item['node_index'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'node_indices': node_indices,
        'label': labels
    }


def create_graph_expert_config(
    dataset_path=None,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device='cuda',
    checkpoint_dir='../../autodl-fs/model',
    hidden_dim=128,
    expert_dim=64,
    num_layers=2,
    dropout=0.3,
    expert_names=['des', 'tweets'],  # 用于聚合节点特征的专家列表
    twibot_dataset=None
):
    """
    创建 Graph Expert 配置

    节点特征从其他专家的表示聚合而来，使用 RGCN 进行图卷积

    Args:
        expert_names: 用于聚合节点特征的专家列表（必须已训练好）
        twibot_dataset: 预加载的Twibot20数据集对象（可选，避免重复加载）

    Returns:
        dict: 包含模型、数据加载器、优化器等的配置字典
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / 'processed_data')

    print(f"\n{'='*60}")
    print(f"配置 Graph Expert (RGCN)")
    print(f"{'='*60}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  使用专家: {expert_names}")

    # 加载数据（如果没有预加载）
    if twibot_dataset is None:
        print("\n加载数据集和图结构...")
        twibot_dataset = Twibot20(root=dataset_path, device=device, process=True, save=True)
    else:
        print("\n使用预加载的数据集和图结构...")

    labels = twibot_dataset.load_labels()
    labels = labels.cpu().numpy()

    # 构建图结构
    edge_index, edge_type = twibot_dataset.build_graph()
    print(f"  图边数: {edge_index.shape[1]}")
    print(f"  边类型数: {len(torch.unique(edge_type))}")

    # 获取训练/验证/测试集索引
    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    print(f"  训练集: {len(train_idx)} 节点")
    print(f"  验证集: {len(val_idx)} 节点")
    print(f"  测试集: {len(test_idx)} 节点")

    # ========== 聚合节点特征：从其他专家提取表示 ==========
    print(f"\n{'='*60}")
    print(f"从其他专家聚合节点特征")
    print(f"{'='*60}")

    from tqdm import tqdm

    # 注意：需要对全部节点（包括支持集）提取特征
    # df_data 包含所有节点（train + val + test + support）
    num_all_nodes = len(twibot_dataset.df_data)
    print(f"  总节点数（含支持集）: {num_all_nodes}")

    all_node_indices = list(range(num_all_nodes)) # 所有节点索引

    # 定义缓存文件路径
    cache_dir = Path(dataset_path) / 'graph_expert_cache'
    cache_dir.mkdir(exist_ok=True)

    # 生成缓存文件名（包含专家列表信息）
    expert_names_str = '_'.join(sorted(expert_names))
    aggregated_features_cache = cache_dir / f'aggregated_node_features_{expert_names_str}.pt'

    # 检查是否存在缓存文件
    if aggregated_features_cache.exists():
        print(f"\n  ✓ 发现缓存的聚合特征，直接加载...")
        print(f"    缓存路径: {aggregated_features_cache}")
        initial_node_features = torch.load(aggregated_features_cache, map_location='cpu')
        print(f"  ✓ 加载完成，节点特征形状: {initial_node_features.shape}")
    else:
        print(f"\n  未找到缓存，开始提取专家特征...")

        # 存储所有节点的专家表示
        all_expert_embeddings = []  # List of [num_all_nodes, 64]

        for expert_name in expert_names:
            # 检查单个专家的缓存
            expert_cache_file = cache_dir / f'{expert_name}_node_features.pt'

            if expert_cache_file.exists():
                print(f"\n  ✓ 加载 {expert_name} 专家的缓存特征...")
                print(f"    缓存路径: {expert_cache_file}")
                cached_data = torch.load(expert_cache_file, map_location='cpu')
                embeddings = cached_data['embeddings']
                mask = cached_data['mask']
                print(f"    有效样本: {mask.sum().item()}/{len(mask)}")
            else:
                print(f"\n  提取 {expert_name} 专家特征...")

                # 提取该专家对所有节点的表示（包括支持集）
                # 使用本地函数提取特征
                embeddings, mask = _extract_expert_features_with_mask(
                    expert_name, all_node_indices, dataset_path, checkpoint_dir, device,
                    twibot_dataset=twibot_dataset  # 传入已加载的数据集，避免重复加载
                )

                # 保存单个专家的特征到缓存
                print(f"    保存 {expert_name} 特征到缓存...")
                torch.save({
                    'embeddings': embeddings.cpu(),
                    'mask': mask.cpu()
                }, expert_cache_file)
                print(f"    ✓ 已保存到: {expert_cache_file}")

            all_expert_embeddings.append(embeddings)  # [num_all_nodes, 64]

        # 聚合所有专家的表示：简单拼接后通过线性层
        print(f"\n  聚合 {len(expert_names)} 个专家的特征...")
        stacked_embeddings = torch.stack(all_expert_embeddings, dim=1)  # [num_all_nodes, num_experts, 64]

        # 方式1：平均池化
        initial_node_features = torch.mean(stacked_embeddings, dim=1)  # [num_all_nodes, 64]

        # 方式2：拼接（如果想用这个，需要修改 input_dim）
        # initial_node_features = stacked_embeddings.view(num_all_nodes, -1)  # [num_all_nodes, num_experts*64]

        print(f"  初始节点特征形状: {initial_node_features.shape}")

        # 保存聚合后的特征
        print(f"\n  保存聚合特征到缓存...")
        torch.save(initial_node_features.cpu(), aggregated_features_cache)
        print(f"  ✓ 已保存到: {aggregated_features_cache}")

    # 创建数据集和数据加载器（只用带标签的节点）
    print("\n创建数据加载器...")
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_dataset = GraphDataset(train_idx, train_labels)
    val_dataset = GraphDataset(val_idx, val_labels)
    test_dataset = GraphDataset(test_idx, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_fn)

    # 初始化模型
    print("\n初始化 Graph Expert 模型...")
    model = GraphExpert(
        num_nodes=num_all_nodes,
        initial_node_features=initial_node_features,
        num_relations=2,  # following (0) 和 follower (1)
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    ).to(device)

    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # 数据提取函数
    def extract_fn(batch, device):
        node_indices = batch['node_indices'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        # 图结构是全局的，不随 batch 变化
        return (node_indices, edge_index, edge_type), labels

    return {
        'name': 'graph',
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'extract_fn': extract_fn,
        'edge_index': edge_index,  # 保存图结构供后续使用
        'edge_type': edge_type
    }


# ==================== 配置注册表 ====================

EXPERT_CONFIGS = {
    'des': create_des_expert_config,
    'tweets': create_tweets_expert_config,
    'graph': create_graph_expert_config,
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

