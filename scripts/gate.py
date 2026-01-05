"""
门控机制 - 基于节点度数的专家路由
- 度数 <= degree_threshold 的节点: 使用 HierarchicalFusionExpert（特征融合专家）
- 度数 > degree_threshold 的节点: 使用 GraphExpert（图专家）

包含完整的训练、验证、测试功能
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import os
import argparse

# 获取项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加到 Python 路径
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import Twibot20
from src.metrics import update_binary_counts, compute_binary_f1
from scripts.feature_fusion import HierarchicalFusionExpert
from src.model import GraphExpert
from scripts.iso_and_nonisol import analyze_node_degrees


# 门控数据集
class GatedDataset(Dataset):
    def __init__(self, node_indices, labels, degree,
                 cat_embeddings, num_embeddings, des_embeddings, post_embeddings,
                 des_mask, post_mask, mode='train', degree_threshold=20):
        """
        Args:
            node_indices: 节点索引列表
            labels: 标签列表
            degree: 节点度数数组 [num_nodes]
            cat_embeddings: 类别专家嵌入 [num_nodes, 64]
            num_embeddings: 数值专家嵌入 [num_nodes, 64]
            des_embeddings: 描述专家嵌入 [num_nodes, 64]
            post_embeddings: 推文专家嵌入 [num_nodes, 64]
            des_mask: 描述有效性掩码 [num_nodes]
            post_mask: 推文有效性掩码 [num_nodes]
            mode: 'train' | 'val' | 'test'
            degree_threshold: int, 将节点划为孤立的度数阈值（<= threshold视为孤立）
        """
        self.mode = mode
        self.node_indices = node_indices
        self.labels = labels
        self.degree = degree
        self.degree_threshold = degree_threshold

        # 特征嵌入
        self.cat_embeddings = cat_embeddings
        self.num_embeddings = num_embeddings
        self.des_embeddings = des_embeddings
        self.post_embeddings = post_embeddings
        self.des_mask = des_mask.bool() if isinstance(des_mask, torch.Tensor) else torch.tensor(des_mask,
                                                                                                dtype=torch.bool)
        self.post_mask = post_mask.bool() if isinstance(post_mask, torch.Tensor) else torch.tensor(post_mask,
                                                                                                   dtype=torch.bool)

        # 统计节点度数分布
        degrees_of_nodes = degree[node_indices]
        isolated_mask = degrees_of_nodes <= self.degree_threshold
        num_isolated = isolated_mask.sum().item()
        num_non_isolated = len(node_indices) - num_isolated

        print(f"  [{mode}集] 样本数量: {len(node_indices)}")
        print(f"    使用孤立阈值: {self.degree_threshold}")
        print(f"    孤立节点（度<={self.degree_threshold}）: {num_isolated} ({num_isolated / len(node_indices) * 100:.1f}%)")
        print(f"    非孤立节点（度>{self.degree_threshold}）: {num_non_isolated} ({num_non_isolated / len(node_indices) * 100:.1f}%)")

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = self.node_indices[idx]
        return {
            'node_index': torch.tensor(node_idx, dtype=torch.long),
            'degree': self.degree[node_idx],
            'cat_repr': self.cat_embeddings[node_idx],
            'num_repr': self.num_embeddings[node_idx],
            'des_repr': self.des_embeddings[node_idx],
            'post_repr': self.post_embeddings[node_idx],
            'des_mask': self.des_mask[node_idx],
            'post_mask': self.post_mask[node_idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# 门控专家模型
class GatedExpertModel(nn.Module):
    """
    门控专家模型 - 基于节点度数路由

    路由策略:
        - 度数 <= threshold: 使用 HierarchicalFusionExpert（特征融合）
        - 度数 > threshold: 使用 GraphExpert（图神经网络）
    """

    def __init__(self, fusion_expert, graph_expert, edge_index, edge_type,
                 degree_threshold=20, device='cuda'):
        """
        Args:
            fusion_expert: HierarchicalFusionExpert 实例
            graph_expert: GraphExpert 实例
            edge_index: 图边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            degree_threshold: 度数阈值（默认1）
            device: 设备
        """
        super(GatedExpertModel, self).__init__()

        self.fusion_expert = fusion_expert
        self.graph_expert = graph_expert
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.degree_threshold = degree_threshold
        self.device = device

        # 统计本次分别用了多少比例的混合专家/图专家
        self.stats = {
            'fusion_count': 0,
            'graph_count': 0
        }

    def forward(self, node_indices, degree, cat_repr, num_repr, des_repr, post_repr,
                des_mask, post_mask):
        """
        前向传播 - 根据节点度数路由到不同专家

        Args:
            node_indices: [batch_size] 节点索引
            degree: [batch_size] 节点度数
            cat_repr: [batch_size, 64] 类别表示
            num_repr: [batch_size, 64] 数值表示
            des_repr: [batch_size, 64] 描述表示
            post_repr: [batch_size, 64] 推文表示
            des_mask: [batch_size] 描述有效性掩码
            post_mask: [batch_size] 推文有效性掩码

        Returns:
            expert_repr: [batch_size, 64] 专家表示（用于保存嵌入）
            bot_prob: [batch_size, 1] bot概率
        """
        batch_size = node_indices.shape[0]
        bot_prob = torch.zeros(batch_size, 1, device=self.device) # 当前批次的预测概率
        expert_repr = torch.zeros(batch_size, 64, device=self.device) # 当前批次的专家表示

        # 分离孤立节点和非孤立节点
        isolated_mask = degree <= self.degree_threshold
        non_isolated_mask = ~isolated_mask

        # 统计路由数量
        num_isolated = isolated_mask.sum().item()
        num_non_isolated = non_isolated_mask.sum().item()
        self.stats['fusion_count'] += num_isolated
        self.stats['graph_count'] += num_non_isolated

        # 处理孤立节点 - 使用特征融合专家
        if num_isolated > 0:
            isolated_indices = isolated_mask.nonzero(as_tuple=True)[0]
            fusion_repr, fusion_prob = self.fusion_expert(
                cat_repr[isolated_indices],
                num_repr[isolated_indices],
                des_repr[isolated_indices],
                post_repr[isolated_indices],
                des_mask[isolated_indices],
                post_mask[isolated_indices]
            )
            bot_prob[isolated_indices] = fusion_prob
            expert_repr[isolated_indices] = fusion_repr

        # 处理非孤立节点 - 使用图专家
        if num_non_isolated > 0:
            non_isolated_indices = non_isolated_mask.nonzero(as_tuple=True)[0]
            graph_node_indices = node_indices[non_isolated_indices]
            graph_repr, graph_prob = self.graph_expert(
                graph_node_indices,
                self.edge_index,
                self.edge_type
            )
            bot_prob[non_isolated_indices] = graph_prob
            expert_repr[non_isolated_indices] = graph_repr

        return expert_repr, bot_prob

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {'fusion_count': 0, 'graph_count': 0}

    def print_stats(self):
        """打印路由统计"""
        total = self.stats['fusion_count'] + self.stats['graph_count']
        if total > 0:
            print(f"\n门控路由统计:")
            print(f"  特征融合专家: {self.stats['fusion_count']} ({self.stats['fusion_count'] / total * 100:.1f}%)")
            print(f"  图专家: {self.stats['graph_count']} ({self.stats['graph_count'] / total * 100:.1f}%)")

# 门控专家训练器
class GatedExpertTrainer:
    """门控专家训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, device, checkpoint_dir,
                 early_stopping_patience=5, max_grad_norm=1.0):
        """
        Args:
            model: GatedExpertModel 实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            checkpoint_dir: 检查点保存目录
            early_stopping_patience: 早停耐心值
            max_grad_norm: 梯度裁剪阈值
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0
        self.history = {'train': [], 'val': [], 'test': {}}

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        self.model.reset_stats()

        pbar = tqdm(self.train_loader, desc=f"[GATE] Epoch {epoch} [Train]")
        for batch in pbar:
            node_indices = batch['node_index'].to(self.device)
            degree = batch['degree'].to(self.device)
            cat_repr = batch['cat_repr'].to(self.device)
            num_repr = batch['num_repr'].to(self.device)
            des_repr = batch['des_repr'].to(self.device)
            post_repr = batch['post_repr'].to(self.device)
            des_mask = batch['des_mask'].to(self.device)
            post_mask = batch['post_mask'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            _, bot_prob = self.model(
                node_indices, degree, cat_repr, num_repr,
                des_repr, post_repr, des_mask, post_mask
            )

            loss = self.criterion(bot_prob, labels)
            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            predictions = (bot_prob > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            update_binary_counts(predictions, labels, counts)
            _, _, f1_running = compute_binary_f1(counts)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / total:.4f}',
                'f1': f'{f1_running:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 不再每轮打印详细统计信息，在tqdm进度条中已显示关键指标

        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        self.model.reset_stats()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[GATE] Epoch {epoch} [Val]")
            for batch in pbar:
                node_indices = batch['node_index'].to(self.device)
                degree = batch['degree'].to(self.device)
                cat_repr = batch['cat_repr'].to(self.device)
                num_repr = batch['num_repr'].to(self.device)
                des_repr = batch['des_repr'].to(self.device)
                post_repr = batch['post_repr'].to(self.device)
                des_mask = batch['des_mask'].to(self.device)
                post_mask = batch['post_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)

                _, bot_prob = self.model(
                    node_indices, degree, cat_repr, num_repr,
                    des_repr, post_repr, des_mask, post_mask
                )

                loss = self.criterion(bot_prob, labels)

                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                _, _, f1_running = compute_binary_f1(counts)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 不再每轮打印路由统计（在训练开始时已打印一次）

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_val_f1 = f1
            self.save_checkpoint('best', epoch, {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1})
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_loss:.4f}, Val F1: {f1:.4f})")
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter <= self.early_stopping_patience:
                print(f"  早停计数器: {self.early_stopping_counter}/{self.early_stopping_patience}")

        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def test(self):
        """测试"""
        best_checkpoint_path = self.checkpoint_dir / 'gated_expert_best.pth'
        if best_checkpoint_path.exists():
            print(f"加载最佳模型: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("警告: 未找到最佳模型检查点，使用当前模型进行测试")

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        self.model.reset_stats()

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"[GATE] [Test]")
            for batch in pbar:
                node_indices = batch['node_index'].to(self.device)
                degree = batch['degree'].to(self.device)
                cat_repr = batch['cat_repr'].to(self.device)
                num_repr = batch['num_repr'].to(self.device)
                des_repr = batch['des_repr'].to(self.device)
                post_repr = batch['post_repr'].to(self.device)
                des_mask = batch['des_mask'].to(self.device)
                post_mask = batch['post_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)

                _, bot_prob = self.model(
                    node_indices, degree, cat_repr, num_repr,
                    des_repr, post_repr, des_mask, post_mask
                )

                loss = self.criterion(bot_prob, labels)

                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                _, _, f1_running = compute_binary_f1(counts)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })

        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 测试阶段打印路由统计
        self.model.print_stats()

        test_metrics = {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"\n{'=' * 60}")
        print(f"测试结果:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"{'=' * 60}\n")

        return test_metrics

    def _evaluate_split(self, data_loader, split_name='Split'):
        """
        评估单个数据集分割（用于无训练参数的情况）

        Args:
            data_loader: 数据加载器
            split_name: 数据集名称（'Train', 'Val', 'Test'）

        Returns:
            dict: 包含评估指标的字典
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        self.model.reset_stats()

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"[GATE] Evaluating {split_name}")
            for batch in pbar:
                node_indices = batch['node_index'].to(self.device)
                degree = batch['degree'].to(self.device)
                cat_repr = batch['cat_repr'].to(self.device)
                num_repr = batch['num_repr'].to(self.device)
                des_repr = batch['des_repr'].to(self.device)
                post_repr = batch['post_repr'].to(self.device)
                des_mask = batch['des_mask'].to(self.device)
                post_mask = batch['post_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)

                _, bot_prob = self.model(
                    node_indices, degree, cat_repr, num_repr,
                    des_repr, post_repr, des_mask, post_mask
                )

                loss = self.criterion(bot_prob, labels)

                total_loss += loss.item()
                predictions = (bot_prob > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                _, _, f1_running = compute_binary_f1(counts)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })

        avg_loss = total_loss / len(data_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 打印路由统计
        self.model.print_stats()

        metrics = {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"\n{split_name} 结果:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

        return metrics

    def save_checkpoint(self, name, epoch, metrics):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f'gated_expert_{name}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

    def fit(self, num_epochs):
        """训练模型"""
        print(f"\n{'=' * 60}")
        print(f"开始训练门控专家模型")
        print(f"  训练轮数: {num_epochs}")
        print(f"  早停耐心值: {self.early_stopping_patience}")
        print(f"{'=' * 60}\n")

        # 在第一轮训练前统计并打印路由信息（只打印一次）
        print("正在统计路由信息...")
        self.model.reset_stats()
        self.model.eval()
        with torch.no_grad():
            for batch in self.train_loader:
                node_indices = batch['node_index'].to(self.device)
                degree = batch['degree'].to(self.device)
                cat_repr = batch['cat_repr'].to(self.device)
                num_repr = batch['num_repr'].to(self.device)
                des_repr = batch['des_repr'].to(self.device)
                post_repr = batch['post_repr'].to(self.device)
                des_mask = batch['des_mask'].to(self.device)
                post_mask = batch['post_mask'].to(self.device)

                self.model(
                    node_indices, degree, cat_repr, num_repr,
                    des_repr, post_repr, des_mask, post_mask
                )

        self.model.print_stats()
        print()

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.history['train'].append(train_metrics)

            val_metrics = self.validate(epoch)
            self.history['val'].append(val_metrics)

            # 简洁的每轮摘要（tqdm已显示详细进度）
            print(f"Epoch {epoch}/{num_epochs} - Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"{'=' * 60}\n")

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"早停触发！验证loss在{self.early_stopping_patience}轮内未改善")
                break

        print(f"\n{'=' * 60}")
        print(f"训练完成！开始测试...")
        print(f"{'=' * 60}\n")

        test_metrics = self.test()
        self.history['test'] = test_metrics

        return self.history

# 辅助函数
def load_embedding(path):
    """加载嵌入，自动处理各种格式"""
    data = torch.load(path, map_location='cpu')

    if isinstance(data, torch.Tensor):
        return data

    if isinstance(data, dict):
        if 'train' in data and 'val' in data and 'test' in data:
            train_data = data['train']
            val_data = data['val']
            test_data = data['test']

            if isinstance(train_data, dict):
                train_emb = train_data['embeddings']
                val_emb = val_data['embeddings']
                test_emb = test_data['embeddings']
            else:
                train_emb = train_data
                val_emb = val_data
                test_emb = test_data

            if not isinstance(train_emb, torch.Tensor):
                train_emb = torch.tensor(train_emb, dtype=torch.float32)
            if not isinstance(val_emb, torch.Tensor):
                val_emb = torch.tensor(val_emb, dtype=torch.float32)
            if not isinstance(test_emb, torch.Tensor):
                test_emb = torch.tensor(test_emb, dtype=torch.float32)

            full_emb = torch.cat([train_emb, val_emb, test_emb], dim=0)
            return full_emb

        for key in ['embeddings', 'embedding', 'tensor', 'data']:
            if key in data:
                emb = data[key]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                return emb

        print(f"  错误: {path} 是字典，但格式不符合预期")
        print(f"  可用键: {list(data.keys())}")
        raise KeyError(f"无法从字典中提取嵌入数据")

    raise TypeError(f"不支持的嵌入文件格式: {type(data)}")

def load_mask_from_embedding(path):
    """从嵌入文件中提取 mask（如果存在）"""
    data = torch.load(path, map_location='cpu')

    if isinstance(data, dict) and 'train' in data:
        train_data = data['train']
        val_data = data['val']
        test_data = data['test']

        if isinstance(train_data, dict) and 'mask' in train_data:
            train_mask = train_data['mask']
            val_mask = val_data['mask']
            test_mask = test_data['mask']

            if not isinstance(train_mask, torch.Tensor):
                train_mask = torch.tensor(train_mask, dtype=torch.bool)
            if not isinstance(val_mask, torch.Tensor):
                val_mask = torch.tensor(val_mask, dtype=torch.bool)
            if not isinstance(test_mask, torch.Tensor):
                test_mask = torch.tensor(test_mask, dtype=torch.bool)

            full_mask = torch.cat([train_mask, val_mask, test_mask], dim=0)
            return full_mask

    return None


def main():
    """主函数 - 训练和评估门控专家模型"""

    # ========== 解析命令行参数 ==========
    parser = argparse.ArgumentParser(description='门控专家训练器 - 基于节点度数的专家路由')
    parser.add_argument('--degree_threshold', type=int, default=20,
                        help='节点度数阈值，度数<=阈值使用融合专家，度数>阈值使用图专家 (默认: 1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='早停耐心值 (默认: 5)')
    args = parser.parse_args()

    # ========== 配置 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 路径配置
    embedding_dir = '../../autodl-fs/labeled_embedding'
    checkpoint_dir = '../../autodl-fs/model'
    dataset_path = './processed_data'

    # 训练配置（从命令行参数或默认值）
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = 0.01
    num_epochs = args.num_epochs
    early_stopping_patience = args.early_stopping_patience
    max_grad_norm = 1.0
    degree_threshold = args.degree_threshold

    print(f"训练配置:")
    print(f"  度数阈值: {degree_threshold}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  早停耐心值: {early_stopping_patience}\n")

    # ========== 加载数据集 ==========
    print("=" * 60)
    print("1. 加载数据集")
    print("=" * 60)

    twibot_dataset = Twibot20(root=dataset_path, device=device, process=False, save=True)

    train_idx, val_idx, test_idx = twibot_dataset.train_val_test_mask()
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    print(f"  训练集: {len(train_idx)} 节点")
    print(f"  验证集: {len(val_idx)} 节点")
    print(f"  测试集: {len(test_idx)} 节点")

    labels = twibot_dataset.load_labels().cpu().numpy()
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    # ========== 加载图结构和节点度数 ==========
    print("\n" + "=" * 60)
    print("2. 加载图结构和分析节点度数")
    print("=" * 60)

    edge_index, edge_type = twibot_dataset.build_graph()
    print(f"  图边数: {edge_index.shape[1]}")
    print(f"  边类型数: {len(torch.unique(edge_type))}")

    num_all_nodes = int(edge_index.max().item()) + 1
    num_labeled_nodes = len(labels)

    print(f"  总节点数（含支持集）: {num_all_nodes}")
    print(f"  有标签节点数: {num_labeled_nodes}")

    degree, degree_stats, out_degree, in_degree = analyze_node_degrees(
        edge_index, num_all_nodes, labeled_nodes_count=num_labeled_nodes
    )

    print(f"  有标签节点度数统计:")
    for d in range(6):
        count = len(degree_stats[d])
        print(f"    度为{d}的节点: {count} ({count / num_labeled_nodes * 100:.1f}%)")

    # ========== 加载专家嵌入 ==========
    print("\n" + "=" * 60)
    print("3. 加载专家嵌入")
    print("=" * 60)

    try:
        cat_embeddings = load_embedding(os.path.join(embedding_dir, 'cat_embeddings.pt'))
        num_embeddings = load_embedding(os.path.join(embedding_dir, 'num_embeddings.pt'))
        des_embeddings = load_embedding(os.path.join(embedding_dir, 'des_embeddings.pt'))
        post_embeddings = load_embedding(os.path.join(embedding_dir, 'post_embeddings.pt'))

        print(f"  类别嵌入: {cat_embeddings.shape}")
        print(f"  数值嵌入: {num_embeddings.shape}")
        print(f"  描述嵌入: {des_embeddings.shape}")
        print(f"  推文嵌入: {post_embeddings.shape}")

    except Exception as e:
        print(f"\n❌ 加载专家嵌入失败: {e}")
        print("\n可能的原因：")
        print("1. 专家模型尚未训练完成")
        print("2. embedding 文件路径不正确")
        print("3. embedding 文件格式不兼容")
        print("\n请先运行以下命令训练专家模型：")
        print("  python train_experts.py --expert des,post,cat,num --num_epochs 10")
        return

    # ========== 加载 mask ==========
    try:
        des_mask = load_mask_from_embedding(os.path.join(embedding_dir, 'des_embeddings.pt'))
        post_mask = load_mask_from_embedding(os.path.join(embedding_dir, 'post_embeddings.pt'))

        if des_mask is None:
            des_mask = torch.ones(len(des_embeddings), dtype=torch.bool)
            print(f"  描述掩码: 未找到，假设所有样本有效")
        else:
            print(f"  描述掩码: {des_mask.shape} (有效: {des_mask.sum().item()}/{len(des_mask)})")

        if post_mask is None:
            post_mask = torch.ones(len(post_embeddings), dtype=torch.bool)
            print(f"  推文掩码: 未找到，假设所有样本有效")
        else:
            print(f"  推文掩码: {post_mask.shape} (有效: {post_mask.sum().item()}/{len(post_mask)})")

    except Exception as e:
        print(f"\n⚠️  加载 mask 失败，假设所有样本都有效: {e}")
        des_mask = torch.ones(len(des_embeddings), dtype=torch.bool)
        post_mask = torch.ones(len(post_embeddings), dtype=torch.bool)

    # ========== 创建数据集 ==========
    print("\n" + "=" * 60)
    print("4. 创建门控数据集")
    print("=" * 60)

    train_dataset = GatedDataset(
        train_idx, train_labels, degree,
        cat_embeddings, num_embeddings, des_embeddings, post_embeddings,
        des_mask, post_mask, mode='train', degree_threshold=degree_threshold
    )

    val_dataset = GatedDataset(
        val_idx, val_labels, degree,
        cat_embeddings, num_embeddings, des_embeddings, post_embeddings,
        des_mask, post_mask, mode='val', degree_threshold=degree_threshold
    )

    test_dataset = GatedDataset(
        test_idx, test_labels, degree,
        cat_embeddings, num_embeddings, des_embeddings, post_embeddings,
        des_mask, post_mask, mode='test', degree_threshold=degree_threshold
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 创建模型 ==========
    print("\n" + "=" * 60)
    print("5. 创建门控专家模型")
    print("=" * 60)

    print("\n加载特征融合专家...")
    fusion_expert = HierarchicalFusionExpert(expert_dim=64, dropout=0.2, device=device)
    fusion_checkpoint = torch.load(
        os.path.join(checkpoint_dir, 'hierarchical_fusion_expert_best.pt'),
        map_location=device
    )
    fusion_expert.load_state_dict(fusion_checkpoint['model_state_dict'])
    # 冻结特征融合专家的参数，不在门控训练中更新
    for param in fusion_expert.parameters():
        param.requires_grad = False
    fusion_expert.eval()  # 设置为评估模式
    print("  ✓ 特征融合专家加载成功 (参数已冻结)")

    print("\n加载图专家...")
    print(f"  使用节点数（含支持集）: {num_all_nodes}")
    graph_expert = GraphExpert(
        num_nodes=num_all_nodes,
        embedding_dir='../../autodl-fs/node_embedding',
        num_relations=2,
        embedding_dim=128,
        expert_dim=64,
        dropout=0.3,
        device=device
    )
    graph_checkpoint = torch.load(
        os.path.join(checkpoint_dir, 'graph_expert_best.pt'),
        map_location=device
    )
    graph_expert.load_state_dict(graph_checkpoint['model_state_dict'])
    # 冻结图专家的参数，不在门控训练中更新
    for param in graph_expert.parameters():
        param.requires_grad = False
    graph_expert.eval()  # 设置为评估模式
    print("  ✓ 图专家加载成功 (参数已冻结)")

    print("\n创建门控模型...")
    gated_model = GatedExpertModel(
        fusion_expert=fusion_expert,
        graph_expert=graph_expert,
        edge_index=edge_index,
        edge_type=edge_type,
        degree_threshold=degree_threshold,
        device=device
    ).to(device)
    print(f"  ✓ 门控模型创建成功 (度数阈值: {degree_threshold})")

    # 检查是否有可训练参数
    trainable_params = sum(p.numel() for p in gated_model.parameters() if p.requires_grad)
    print(f"  可训练参数数量: {trainable_params}")
    if trainable_params == 0:
        print(f"  ⚠️  注意: 由于专家模型参数已冻结，门控模型本身无可训练参数")
        print(f"  ⚠️  门控模型仅作为路由机制，直接使用预训练专家的输出")

    # ========== 创建优化器和损失函数 ==========
    print("\n" + "=" * 60)
    print("6. 创建优化器和损失函数")
    print("=" * 60)

    optimizer = AdamW(gated_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    print(f"  优化器: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  损失函数: BCELoss")

    # ========== 创建训练器 ==========
    trainer = GatedExpertTrainer(
        model=gated_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=early_stopping_patience,
        max_grad_norm=max_grad_norm
    )

    # ========== 训练或直接评估 ==========
    if trainable_params == 0:
        print("\n" + "=" * 60)
        print("⚠️  检测到无可训练参数，跳过训练，直接进行评估")
        print("=" * 60)

        # 直接在所有数据集上评估
        print("\n正在统计路由信息...")
        gated_model.reset_stats()
        gated_model.eval()
        with torch.no_grad():
            for batch in train_loader:
                node_indices = batch['node_index'].to(device)
                degree = batch['degree'].to(device)
                cat_repr = batch['cat_repr'].to(device)
                num_repr = batch['num_repr'].to(device)
                des_repr = batch['des_repr'].to(device)
                post_repr = batch['post_repr'].to(device)
                des_mask = batch['des_mask'].to(device)
                post_mask = batch['post_mask'].to(device)

                gated_model(
                    node_indices, degree, cat_repr, num_repr,
                    des_repr, post_repr, des_mask, post_mask
                )

        gated_model.print_stats()

        print("\n" + "=" * 60)
        print("评估训练集")
        print("=" * 60)
        train_metrics = trainer._evaluate_split(train_loader, split_name='Train')

        print("\n" + "=" * 60)
        print("评估验证集")
        print("=" * 60)
        val_metrics = trainer._evaluate_split(val_loader, split_name='Val')

        print("\n" + "=" * 60)
        print("评估测试集")
        print("=" * 60)
        test_metrics = trainer._evaluate_split(test_loader, split_name='Test')

        history = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
    else:
        # ========== 训练和测试 ==========
        history = trainer.fit(num_epochs=num_epochs)

    # ========== 保存训练历史 ==========
    print("\n" + "=" * 60)
    print("保存训练历史")
    print("=" * 60)

    import json
    history_path = os.path.join(checkpoint_dir, f'gated_expert_history_threshold_{degree_threshold}.json')

    # 根据history格式决定如何序列化
    if isinstance(history['train'], list):
        # 训练模式：有多个epoch的历史
        history_serializable = {
            'config': {
                'degree_threshold': degree_threshold,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs
            },
            'train': [
                {k: float(v) for k, v in epoch_metrics.items()}
                for epoch_metrics in history['train']
            ],
            'val': [
                {k: float(v) for k, v in epoch_metrics.items()}
                for epoch_metrics in history['val']
            ],
            'test': {k: float(v) for k, v in history['test'].items()}
        }
    else:
        # 直接评估模式：每个split只有一个结果
        history_serializable = {
            'config': {
                'degree_threshold': degree_threshold,
                'batch_size': batch_size,
                'mode': 'direct_evaluation'
            },
            'train': {k: float(v) for k, v in history['train'].items()},
            'val': {k: float(v) for k, v in history['val'].items()},
            'test': {k: float(v) for k, v in history['test'].items()}
        }

    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=4)

    print(f"  训练历史已保存到: {history_path}")
    print("\n" + "=" * 60)
    print("训练和测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
