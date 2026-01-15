"""
MoE (Mixture of Experts) 模块
- 专家1: Profile Expert (数值+类别+简介)
- 专家2: Tweet Expert (推文)
- 门控网络: 基于元数据动态计算权重
- RGCN: 图结构聚合
- 对比学习: InfoNCE Loss
- 融合预测: 分类输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import sys

# 添加项目路径 - 无论从哪里运行都能正确导入
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from torch_geometric.nn import RGCNConv
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    RGCNConv = None
    _TORCH_GEOMETRIC_AVAILABLE = False

from src.dataset import Twibot20
from src.metrics import update_binary_counts, compute_binary_f1


# ==================== 专家网络 ====================

class ProfileExpert(nn.Module):
    """
    Profile 专家 - 处理 数值+简介
    刻画"它看起来像谁"
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class TweetExpert(nn.Module):
    """
    Tweet 专家 - 处理推文嵌入
    刻画"它在做什么"
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GatingNetwork(nn.Module):
    """
    门控网络 - 根据数值特征动态计算专家权重
    输入: 数值特征
    输出: [w1, w2] (专家权重，和为1)
    """
    def __init__(self, num_dim, num_experts=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, num_features):
        """
        Args:
            num_features: [batch_size, num_dim]
        Returns:
            weights: [batch_size, num_experts] - softmax 权重
        """
        logits = self.gate(num_features)
        # 温度缩放的 softmax
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        return weights


class MoELayer(nn.Module):
    """
    MoE 层 - 混合专家层
    融合 Profile Expert 和 Tweet Expert 的输出
    """
    def __init__(self, profile_dim, tweet_dim, num_dim,
                 hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim

        # 专家网络
        self.profile_expert = ProfileExpert(profile_dim, hidden_dim, output_dim, dropout)
        self.tweet_expert = TweetExpert(tweet_dim, hidden_dim, output_dim, dropout)

        # 门控网络
        self.gating = GatingNetwork(num_dim, num_experts=2,
                                    hidden_dim=hidden_dim // 2, dropout=dropout)

    def forward(self, profile_features, tweet_features, num_features):
        """
        Args:
            profile_features: [batch_size, profile_dim] - 数值+简介拼接
            tweet_features: [batch_size, tweet_dim] - 推文嵌入
            num_features: [batch_size, num_dim] - 数值特征（用于门控）
        Returns:
            h_moe: [batch_size, output_dim] - 融合后的专家向量
            gate_weights: [batch_size, 2] - 门控权重
        """
        # 计算专家输出
        h_profile = self.profile_expert(profile_features)  # [B, output_dim]
        h_tweet = self.tweet_expert(tweet_features)        # [B, output_dim]

        # 计算门控权重
        gate_weights = self.gating(num_features)  # [B, 2]

        # 加权融合
        h_moe = gate_weights[:, 0:1] * h_profile + gate_weights[:, 1:2] * h_tweet

        return h_moe, gate_weights


# ==================== RGCN 模块 ====================

class RGCNModule(nn.Module):
    """
    RGCN 模块 - 图结构聚合
    输入: 节点特征 h_moe
    输出: 结构表示 h_graph
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=64,
                 num_relations=2, num_layers=2, dropout=0.3):
        super().__init__()

        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("需要安装 torch_geometric: pip install torch-geometric")

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层
        self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations=num_relations))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 最后一层
        if num_layers > 1:
            self.convs.append(RGCNConv(hidden_dim, output_dim, num_relations=num_relations))
            self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [num_nodes, input_dim] - 节点特征
            edge_index: [2, num_edges] - 边索引
            edge_type: [num_edges] - 边类型
        Returns:
            h_graph: [num_nodes, output_dim] - 结构表示
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        return x


# ==================== 对比学习模块 ====================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE 对比损失
    计算 h_moe 和 h_graph 的互信息
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_moe, h_graph):
        """
        Args:
            h_moe: [batch_size, dim] - 语义表示
            h_graph: [batch_size, dim] - 结构表示
        Returns:
            loss: scalar - InfoNCE 损失
        """
        batch_size = h_moe.size(0)

        # L2 归一化
        h_moe = F.normalize(h_moe, dim=-1)
        h_graph = F.normalize(h_graph, dim=-1)

        # 计算相似度矩阵
        # 正样本: 对角线元素 (同一节点的 h_moe 和 h_graph)
        # 负样本: 非对角线元素
        sim_matrix = torch.mm(h_moe, h_graph.t()) / self.temperature  # [B, B]

        # 对称 InfoNCE
        labels = torch.arange(batch_size, device=h_moe.device)

        # h_moe -> h_graph
        loss_moe2graph = F.cross_entropy(sim_matrix, labels)

        # h_graph -> h_moe
        loss_graph2moe = F.cross_entropy(sim_matrix.t(), labels)

        return (loss_moe2graph + loss_graph2moe) / 2


# ==================== 融合预测层 ====================

class FusionClassifier(nn.Module):
    """
    融合预测层
    拼接 h_moe 和 h_graph，输出分类结果
    """
    def __init__(self, moe_dim, graph_dim, hidden_dim=128, num_classes=2,
                 dropout=0.3, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            input_dim = moe_dim + graph_dim
        elif fusion_type == 'add':
            assert moe_dim == graph_dim, "add fusion requires same dimensions"
            input_dim = moe_dim
        elif fusion_type == 'gate':
            input_dim = moe_dim
            self.fusion_gate = nn.Sequential(
                nn.Linear(moe_dim + graph_dim, moe_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, h_moe, h_graph):
        """
        Args:
            h_moe: [batch_size, moe_dim]
            h_graph: [batch_size, graph_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        if self.fusion_type == 'concat':
            h_fused = torch.cat([h_moe, h_graph], dim=-1)
        elif self.fusion_type == 'add':
            h_fused = h_moe + h_graph
        elif self.fusion_type == 'gate':
            gate = self.fusion_gate(torch.cat([h_moe, h_graph], dim=-1))
            h_fused = gate * h_moe + (1 - gate) * h_graph

        return self.classifier(h_fused)


# ==================== 完整 MoE-RGCN 模型 ====================

class MoEBotDetector(nn.Module):
    """
    完整的 MoE Bot 检测模型
    1. 特征分组与编码
    2. MoE 专家融合
    3. RGCN 图聚合
    4. 对比学习
    5. 融合分类
    """
    def __init__(self, num_dim, des_dim, tweet_dim,
                 hidden_dim=128, expert_dim=64, num_relations=2,
                 num_rgcn_layers=2, dropout=0.3, fusion_type='concat',
                 contrastive_weight=0.1, temperature=0.07, device='cuda'):
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.expert_dim = expert_dim
        self.contrastive_weight = contrastive_weight

        # 特征编码器（统一到相同维度）
        # 维度对齐层：数值+简介 -> profile特征
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        self.des_encoder = nn.Sequential(
            nn.Linear(des_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        self.tweet_encoder = nn.Sequential(
            nn.Linear(tweet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 编码后维度
        encoded_profile_dim = hidden_dim // 2 + hidden_dim // 2  # = hidden_dim
        encoded_tweet_dim = hidden_dim

        # MoE 层
        self.moe = MoELayer(
            profile_dim=encoded_profile_dim,
            tweet_dim=encoded_tweet_dim,
            num_dim=num_dim,
            hidden_dim=hidden_dim,
            output_dim=expert_dim,
            dropout=dropout
        )

        # RGCN 模块
        self.rgcn = RGCNModule(
            input_dim=expert_dim,
            hidden_dim=expert_dim,
            output_dim=expert_dim,
            num_relations=num_relations,
            num_layers=num_rgcn_layers,
            dropout=dropout
        )

        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

        # 融合分类器
        self.classifier = FusionClassifier(
            moe_dim=expert_dim,
            graph_dim=expert_dim,
            hidden_dim=hidden_dim,
            num_classes=1,  # 二分类使用 1 输出 + sigmoid
            dropout=dropout,
            fusion_type=fusion_type
        )

    def encode_features(self, num_features, des_features, tweet_features):
        """编码各模态特征"""
        h_num = self.num_encoder(num_features)
        h_des = self.des_encoder(des_features)
        h_tweet = self.tweet_encoder(tweet_features)

        # 拼接 profile 特征
        h_profile = torch.cat([h_num, h_des], dim=-1)

        return h_profile, h_tweet

    def forward(self, num_features, des_features, tweet_features,
                edge_index, edge_type, node_indices=None, return_all=False):
        """
        Args:
            num_features: [num_nodes, num_dim] - 数值特征
            des_features: [num_nodes, des_dim] - 简介嵌入
            tweet_features: [num_nodes, tweet_dim] - 推文嵌入
            edge_index: [2, num_edges] - 边索引
            edge_type: [num_edges] - 边类型
            node_indices: [batch_size] - 当前 batch 的节点索引（用于训练）
            return_all: 是否返回所有中间结果
        Returns:
            logits: [batch_size, 1] - 分类 logits
            loss_dict: dict - 包含各项损失（如果 training）
        """
        # 1. 特征编码
        h_profile, h_tweet = self.encode_features(
            num_features, des_features, tweet_features
        )

        # 2. MoE 专家融合（全图）
        h_moe_all, gate_weights_all = self.moe(
            h_profile, h_tweet, num_features
        )

        # 3. RGCN 图聚合（全图）
        h_graph_all = self.rgcn(h_moe_all, edge_index, edge_type)

        # 4. 如果指定了 node_indices，只取 batch 节点
        if node_indices is not None:
            h_moe = h_moe_all[node_indices]
            h_graph = h_graph_all[node_indices]
            gate_weights = gate_weights_all[node_indices]
        else:
            h_moe = h_moe_all
            h_graph = h_graph_all
            gate_weights = gate_weights_all

        # 5. 融合分类
        logits = self.classifier(h_moe, h_graph)

        # 6. 计算对比损失（训练时）
        loss_dict = {}
        if self.training and h_moe.size(0) > 1:
            loss_contrastive = self.contrastive_loss(h_moe, h_graph)
            loss_dict['contrastive'] = loss_contrastive

        if return_all:
            return logits, loss_dict, {
                'h_moe': h_moe,
                'h_graph': h_graph,
                'gate_weights': gate_weights,
                'h_moe_all': h_moe_all,
                'h_graph_all': h_graph_all
            }

        return logits, loss_dict


# ==================== 数据集类 ====================

class MoEDataset(Dataset):
    """MoE 模型数据集"""
    def __init__(self, node_indices, labels):
        self.node_indices = node_indices
        self.labels = labels

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        return {
            'node_index': self.node_indices[idx],
            'label': self.labels[idx]
        }


def collate_moe_fn(batch):
    """MoE 数据集的 collate 函数"""
    return {
        'node_indices': torch.tensor([item['node_index'] for item in batch], dtype=torch.long),
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    }


# ==================== 数据加载器 ====================

class MoEDataLoader:
    """MoE 模型数据加载器"""
    def __init__(self, dataset_path='./processed_data', embedding_dir='../../autodl-fs/labeled_embedding',
                 device='cuda', batch_size=64):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.embedding_dir = Path(embedding_dir)

        print("="*60)
        print("加载 MoE 数据...")
        print("="*60)

        # 加载 Twibot20 数据集
        self.twibot = Twibot20(root=dataset_path, device=self.device, process=True, save=True)

        # 加载标签
        labels_all = self.twibot.load_labels().cpu().numpy()

        # 加载图结构
        edge_index_orig, self.edge_type = self.twibot.build_graph()

        # 获取数据划分
        train_idx_orig, val_idx_orig, test_idx_orig = self.twibot.train_val_test_mask()
        self.train_idx_orig = list(train_idx_orig)
        self.val_idx_orig = list(val_idx_orig)
        self.test_idx_orig = list(test_idx_orig)

        # 创建从原始索引到连续索引的映射
        # 所有有标签的节点按照 train -> val -> test 的顺序重新编号为 0, 1, 2, ...
        all_labeled_orig = self.train_idx_orig + self.val_idx_orig + self.test_idx_orig
        self.orig_to_new_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(all_labeled_orig)}

        # 重映射标签数组
        self.labels = labels_all[all_labeled_orig]

        # 映射后的连续索引
        self.train_idx = [self.orig_to_new_idx[idx] for idx in self.train_idx_orig]
        self.val_idx = [self.orig_to_new_idx[idx] for idx in self.val_idx_orig]
        self.test_idx = [self.orig_to_new_idx[idx] for idx in self.test_idx_orig]

        # 重映射边索引：只保留两端都在有标签节点中的边
        edge_mask = torch.tensor([
            (edge_index_orig[0, i].item() in self.orig_to_new_idx and
             edge_index_orig[1, i].item() in self.orig_to_new_idx)
            for i in range(edge_index_orig.shape[1])
        ], dtype=torch.bool)

        edge_index_filtered = edge_index_orig[:, edge_mask]
        edge_type_filtered = self.edge_type[edge_mask]

        # 映射到新的连续索引
        self.edge_index = torch.stack([
            torch.tensor([self.orig_to_new_idx[idx.item()] for idx in edge_index_filtered[0]], dtype=torch.long),
            torch.tensor([self.orig_to_new_idx[idx.item()] for idx in edge_index_filtered[1]], dtype=torch.long)
        ]).to(self.device)
        self.edge_type = edge_type_filtered.to(self.device)

        # 加载特征嵌入
        self._load_features()

        # 验证边索引的有效性
        num_nodes = len(all_labeled_orig)
        max_edge_idx = self.edge_index.max().item()
        if max_edge_idx >= num_nodes:
            print(f"警告: 边索引最大值 {max_edge_idx} 超过节点数 {num_nodes}！")
            # 过滤掉无效的边
            valid_edge_mask = (self.edge_index[0] < num_nodes) & (self.edge_index[1] < num_nodes)
            self.edge_index = self.edge_index[:, valid_edge_mask]
            self.edge_type = self.edge_type[valid_edge_mask]
            print(f"过滤后边数: {self.edge_index.shape[1]}")

        print(f"训练集: {len(self.train_idx)}, 验证集: {len(self.val_idx)}, 测试集: {len(self.test_idx)}")
        print(f"总节点数: {self.num_features.shape[0]}")
        print(f"边数: {self.edge_index.shape[1]}")

    def _load_features(self):
        """加载各模态特征"""
        # 所有有标签节点的原始索引
        all_labeled_orig = self.train_idx_orig + self.val_idx_orig + self.test_idx_orig

        # 数值特征 - 只取有标签的节点，按照新的顺序排列
        num_features_all = self.twibot.num_prop_preprocess()
        if isinstance(num_features_all, torch.Tensor):
            self.num_features = num_features_all[all_labeled_orig].to(self.device)
        else:
            self.num_features = torch.tensor(num_features_all, dtype=torch.float32)[all_labeled_orig].to(self.device)

        # 尝试加载预训练的嵌入，否则使用占位符
        des_path = self.embedding_dir / 'des_embeddings.pt'
        tweet_path = self.embedding_dir / 'post_embeddings.pt'

        if des_path.exists():
            des_data = torch.load(des_path, map_location='cpu')
            # 合并 train/val/test 嵌入
            self.des_features = self._merge_split_embeddings(des_data, 'embeddings')
            print(f"  加载简介嵌入: {self.des_features.shape}")
        else:
            print(f"  警告: 未找到简介嵌入 {des_path}，使用随机初始化")
            self.des_features = torch.randn(len(self.labels), 64).to(self.device)

        if tweet_path.exists():
            tweet_data = torch.load(tweet_path, map_location='cpu')
            self.tweet_features = self._merge_split_embeddings(tweet_data, 'embeddings')
            print(f"  加载推文嵌入: {self.tweet_features.shape}")
        else:
            print(f"  警告: 未找到推文嵌入 {tweet_path}，使用随机初始化")
            self.tweet_features = torch.randn(len(self.labels), 64).to(self.device)

        # 确保在正确设备上
        self.des_features = self.des_features.to(self.device)
        self.tweet_features = self.tweet_features.to(self.device)

        print(f"  数值特征: {self.num_features.shape}")

    def _merge_split_embeddings(self, data, key='embeddings'):
        """合并 train/val/test 的嵌入，按照train->val->test顺序连续存储"""
        num_labeled = len(self.train_idx) + len(self.val_idx) + len(self.test_idx)

        # 获取嵌入维度
        sample = data.get('train', data)
        if isinstance(sample, dict):
            sample = sample[key]
        embed_dim = sample.shape[-1]

        merged = torch.zeros(num_labeled, embed_dim)

        current_pos = 0
        for split_name in ['train', 'val', 'test']:
            if split_name in data:
                split_data = data[split_name]
                if isinstance(split_data, dict):
                    embeddings = split_data[key]
                else:
                    embeddings = split_data

                # 按顺序连续填充
                split_size = embeddings.shape[0]
                merged[current_pos:current_pos + split_size] = embeddings
                current_pos += split_size

        return merged

    def get_dataloaders(self):
        """获取 DataLoader"""
        train_ds = MoEDataset(self.train_idx, self.labels[self.train_idx])
        val_ds = MoEDataset(self.val_idx, self.labels[self.val_idx])
        test_ds = MoEDataset(self.test_idx, self.labels[self.test_idx])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_moe_fn)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, collate_fn=collate_moe_fn)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, collate_fn=collate_moe_fn)

        return train_loader, val_loader, test_loader

    def get_feature_dims(self):
        """获取各特征维度"""
        return {
            'num_dim': self.num_features.shape[1],
            'des_dim': self.des_features.shape[1],
            'tweet_dim': self.tweet_features.shape[1]
        }

    def get_graph_data(self):
        """获取图结构数据"""
        return self.edge_index, self.edge_type

    def get_all_features(self):
        """获取所有特征（用于模型前向传播）"""
        return (self.num_features, self.des_features, self.tweet_features)


# ==================== 训练器 ====================

class MoETrainer:
    """MoE 模型训练器"""
    def __init__(self, model, data_loader, learning_rate=1e-3, weight_decay=0.01,
                 contrastive_weight=0.1, checkpoint_dir='../../autodl-fs/model',
                 device='cuda', max_grad_norm=1.0, early_stopping_patience=10):
        self.model = model
        self.data_loader = data_loader
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.patience = early_stopping_patience
        self.contrastive_weight = contrastive_weight

        # 优化器
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        # 获取数据
        self.train_loader, self.val_loader, self.test_loader = data_loader.get_dataloaders()
        self.edge_index, self.edge_type = data_loader.get_graph_data()
        self.all_features = data_loader.get_all_features()

        # 训练状态
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.history = {'train': [], 'val': [], 'test': {}}

    def _run_epoch(self, loader, mode='train', epoch=0):
        """运行一个 epoch"""
        is_train = (mode == 'train')
        self.model.train() if is_train else self.model.eval()

        total_loss = 0
        total_cls_loss = 0
        total_con_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        ctx = torch.enable_grad() if is_train else torch.no_grad()

        with ctx:
            desc = f"Epoch {epoch} [{mode.capitalize()}]" if epoch else f"{mode.capitalize()}"
            pbar = tqdm(loader, desc=desc)

            for batch in pbar:
                node_indices = batch['node_indices'].to(self.device)
                labels = batch['labels'].to(self.device).unsqueeze(1)

                if is_train:
                    self.optimizer.zero_grad()

                # 前向传播
                logits, loss_dict = self.model(
                    *self.all_features,
                    self.edge_index, self.edge_type,
                    node_indices=node_indices
                )

                # 分类损失
                cls_loss = self.criterion(logits, labels)

                # 对比损失
                con_loss = loss_dict.get('contrastive', torch.tensor(0.0, device=self.device))

                # 总损失
                loss = cls_loss + self.contrastive_weight * con_loss

                if is_train:
                    loss.backward()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # 统计
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_con_loss += con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                update_binary_counts(preds, labels, counts)

                _, _, f1_run = compute_binary_f1(counts)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls': f'{cls_loss.item():.4f}',
                    'con': f'{con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss:.4f}',
                    'f1': f'{f1_run:.4f}'
                })

        avg_loss = total_loss / len(loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        return {
            'loss': avg_loss,
            'cls_loss': total_cls_loss / len(loader),
            'con_loss': total_con_loss / len(loader),
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_epoch(self, epoch):
        return self._run_epoch(self.train_loader, 'train', epoch)

    def validate(self, epoch):
        metrics = self._run_epoch(self.val_loader, 'val', epoch)

        if metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = metrics['f1']
            self._save_checkpoint('best', epoch, metrics)
            print(f"  ✓ 保存最佳模型 (Val F1: {metrics['f1']:.4f})")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"  早停计数: {self.patience_counter}/{self.patience}")

        return metrics

    def test(self):
        print("\n测试最佳模型...")
        self._load_checkpoint('best')
        return self._run_epoch(self.test_loader, 'test')

    def train(self, num_epochs, scheduler=None):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("开始训练 MoE Bot Detector")
        print(f"{'='*60}")

        if scheduler is None:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        for epoch in range(1, num_epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            self.history['train'].append(train_m)
            self.history['val'].append(val_m)

            scheduler.step()

            print(f"Epoch {epoch}: Train F1={train_m['f1']:.4f}, Val F1={val_m['f1']:.4f}, "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")

            if self.patience_counter >= self.patience:
                print(f"早停触发！")
                break

        # 测试
        test_m = self.test()
        self.history['test'] = test_m

        print(f"\n{'='*60}")
        print("测试结果:")
        print(f"{'='*60}")
        print(f"  Accuracy:  {test_m['acc']:.4f}")
        print(f"  F1 Score:  {test_m['f1']:.4f}")
        print(f"  Precision: {test_m['precision']:.4f}")
        print(f"  Recall:    {test_m['recall']:.4f}")

        self._save_checkpoint('final', num_epochs, test_m)

        return self.history

    def _save_checkpoint(self, name, epoch, metrics):
        path = self.checkpoint_dir / f"moe_bot_detector_{name}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics
        }, path)

    def _load_checkpoint(self, name):
        path = self.checkpoint_dir / f"moe_bot_detector_{name}.pt"
        if not path.exists():
            print(f"警告: 检查点 {path} 不存在")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])


# ==================== 分析工具 ====================

def analyze_gate_weights(model, data_loader, device='cuda'):
    """分析门控权重分布"""
    model.eval()
    all_features = data_loader.get_all_features()
    edge_index, edge_type = data_loader.get_graph_data()
    labels = data_loader.labels

    with torch.no_grad():
        _, _, info = model(
            *all_features, edge_index, edge_type,
            node_indices=None, return_all=True
        )

    gate_weights = info['gate_weights'].cpu().numpy()

    # 按类别分析
    bot_mask = labels == 1
    human_mask = labels == 0

    print("\n" + "="*60)
    print("门控权重分析")
    print("="*60)
    print(f"Profile Expert 权重 (平均):")
    print(f"  Bot:   {gate_weights[bot_mask, 0].mean():.4f} ± {gate_weights[bot_mask, 0].std():.4f}")
    print(f"  Human: {gate_weights[human_mask, 0].mean():.4f} ± {gate_weights[human_mask, 0].std():.4f}")
    print(f"Tweet Expert 权重 (平均):")
    print(f"  Bot:   {gate_weights[bot_mask, 1].mean():.4f} ± {gate_weights[bot_mask, 1].std():.4f}")
    print(f"  Human: {gate_weights[human_mask, 1].mean():.4f} ± {gate_weights[human_mask, 1].std():.4f}")

    return gate_weights


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='训练 MoE Bot 检测模型')
    parser.add_argument('--dataset_path', type=str, default=str(PROJECT_ROOT / 'processed_data'))
    parser.add_argument('--embedding_dir', type=str, default='../../autodl-fs/labeled_embedding')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--expert_dim', type=int, default=64)
    parser.add_argument('--num_rgcn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'add', 'gate'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--analyze_gates', action='store_true')
    args = parser.parse_args()

    # 设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    data_loader = MoEDataLoader(
        dataset_path=args.dataset_path,
        embedding_dir=args.embedding_dir,
        device=device,
        batch_size=args.batch_size
    )

    # 获取特征维度
    feat_dims = data_loader.get_feature_dims()
    print(f"\n特征维度: {feat_dims}")

    # 创建模型
    model = MoEBotDetector(
        num_dim=feat_dims['num_dim'],
        des_dim=feat_dims['des_dim'],
        tweet_dim=feat_dims['tweet_dim'],
        hidden_dim=args.hidden_dim,
        expert_dim=args.expert_dim,
        num_relations=2,
        num_rgcn_layers=args.num_rgcn_layers,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        device=device
    ).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 创建训练器
    trainer = MoETrainer(
        model=model,
        data_loader=data_loader,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        early_stopping_patience=args.early_stopping
    )

    # 训练
    history = trainer.train(num_epochs=args.num_epochs)

    # 分析门控权重
    if args.analyze_gates:
        analyze_gate_weights(model, data_loader, device)

    print("\n✓ 训练完成!")
    return history


if __name__ == '__main__':
    main()

