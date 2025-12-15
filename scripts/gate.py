"""
Expert Importance Gating Network
用于聚合多个专家模型（cat、num、graph）的预测结果

参考 BotBuster 论文的方法:
1. 将各专家的 64 维表示向量拼接
2. 通过两层 MLP + tanh 激活
3. 最后 softmax 输出归一化权重
4. 加权聚合各专家的预测概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import update_binary_counts, compute_binary_f1


class MoEGate(nn.Module):
    """
    Expert Importance Gating Network

    输入: 三个专家（cat, num, graph）的 64 维表示向量拼接 -> 192 维
    输出: 3 维权重向量（softmax 归一化，和为 1）

    网络结构（参考 BotBuster）:
        - 两层 MLP
        - tanh 激活函数
        - softmax 输出层
    """

    def __init__(self, expert_dim=64, num_experts=3, hidden_dim=64, dropout=0.2):
        """
        Args:
            expert_dim: 每个专家的表示维度（默认 64）
            num_experts: 专家数量（默认 3: cat, num, graph）
            hidden_dim: 隐藏层维度
            dropout: dropout 比例
        """
        super(MoEGate, self).__init__()

        self.expert_dim = expert_dim
        self.num_experts = num_experts
        input_dim = expert_dim * num_experts  # 64 * 3 = 192

        # 两层 MLP + tanh 激活
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
            # softmax 在 forward 中单独处理
        )

        # 专家名称（用于输出）
        self.expert_names = ['cat', 'num', 'graph']

    def forward(self, h_cat, h_num, h_graph):
        """
        Args:
            h_cat: [batch_size, 64] - Cat Expert 的表示向量
            h_num: [batch_size, 64] - Num Expert 的表示向量
            h_graph: [batch_size, 64] - Graph Expert 的表示向量

        Returns:
            weights: [batch_size, 3] - 各专家的权重（softmax 归一化）
                     weights[:, 0] -> α_cat
                     weights[:, 1] -> α_num
                     weights[:, 2] -> α_graph
        """
        # 拼接三个专家的表示向量
        x = torch.cat([h_cat, h_num, h_graph], dim=-1)  # [batch_size, 192]

        # 门控网络
        logits = self.gate(x )  # [batch_size, 3]

        # softmax 归一化
        weights = F.softmax(logits, dim=-1)  # [batch_size, 3]

        return weights

    def get_expert_weights(self, h_cat, h_num, h_graph):
        """
        获取各专家的权重（带名称）

        Returns:
            dict: {'cat': α_cat, 'num': α_num, 'graph': α_graph}
        """
        weights = self.forward(h_cat, h_num, h_graph)
        return {
            'cat': weights[:, 0],
            'num': weights[:, 1],
            'graph': weights[:, 2]
        }


class MoEEnsemble(nn.Module):
    """
    完整的 MoE 集成模型（严格按照 BotBuster 论文实现）

    门控网络输出权重，直接加权聚合各专家的预测概率:
        p_final = α_cat * p_cat + α_num * p_num + α_graph * p_graph
    """

    def __init__(self, expert_dim=64, num_experts=3, hidden_dim=64, dropout=0.2):
        super(MoEEnsemble, self).__init__()

        self.expert_dim = expert_dim
        self.num_experts = num_experts

        # 门控网络：输入专家表示，输出权重
        self.gate = MoEGate(expert_dim, num_experts, hidden_dim, dropout)

    def forward(self, h_cat, h_num, h_graph, p_cat, p_num, p_graph):
        """
        严格按照论文方式：用权重加权各专家的预测概率

        Args:
            h_cat, h_num, h_graph: [batch_size, 64] - 各专家的表示向量（用于计算权重）
            p_cat, p_num, p_graph: [batch_size, 1] - 各专家的预测概率

        Returns:
            p_final: [batch_size, 1] - 最终预测概率
            weights: [batch_size, 3] - 各专家权重 [α_cat, α_num, α_graph]
        """
        # 获取门控权重（基于专家表示向量）
        weights = self.gate(h_cat, h_num, h_graph)  # [batch_size, 3]

        # 加权聚合各专家的预测概率
        # p_final = α_cat * p_cat + α_num * p_num + α_graph * p_graph
        p_final = (weights[:, 0:1] * p_cat +
                   weights[:, 1:2] * p_num +
                   weights[:, 2:3] * p_graph)  # [batch_size, 1]

        return p_final, weights


# ==================== 数据集 ====================

class ExpertEmbeddingDataset(Dataset):
    """
    加载各专家的预训练 embedding 和预测概率数据集

    从 ../../autodl-fs/labeled_embedding/ 目录加载:
        - cat_embeddings.pt (包含 embeddings, probs, labels)
        - num_embeddings.pt
        - graph_embeddings.pt
    """

    def __init__(self, split='train', embedding_dir='../../autodl-fs/labeled_embedding'):
        """
        Args:
            split: 'train', 'val', 或 'test'
            embedding_dir: embedding 文件所在目录
        """
        self.split = split
        embedding_path = Path(embedding_dir)

        # 加载三个专家的 embedding
        cat_data = torch.load(embedding_path / 'cat_embeddings.pt', map_location='cpu')
        num_data = torch.load(embedding_path / 'num_embeddings.pt', map_location='cpu')
        graph_data = torch.load(embedding_path / 'graph_embeddings.pt', map_location='cpu')

        # 获取对应 split 的数据 - embedding
        self.h_cat = cat_data[split]['embeddings']  # [N, 64]
        self.h_num = num_data[split]['embeddings']  # [N, 64]
        self.h_graph = graph_data[split]['embeddings']  # [N, 64]

        # 获取对应 split 的数据 - 预测概率
        self.p_cat = cat_data[split]['probs']  # [N, 1]
        self.p_num = num_data[split]['probs']  # [N, 1]
        self.p_graph = graph_data[split]['probs']  # [N, 1]

        self.labels = cat_data[split]['labels']  # [N, 1]

        # 验证数据一致性
        assert len(self.h_cat) == len(self.h_num) == len(self.h_graph) == len(self.labels), \
            f"数据长度不一致: cat={len(self.h_cat)}, num={len(self.h_num)}, graph={len(self.h_graph)}, labels={len(self.labels)}"

        print(f"[{split.upper()}] 加载 {len(self.labels)} 个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'h_cat': self.h_cat[idx],
            'h_num': self.h_num[idx],
            'h_graph': self.h_graph[idx],
            'p_cat': self.p_cat[idx],
            'p_num': self.p_num[idx],
            'p_graph': self.p_graph[idx],
            'label': self.labels[idx]
        }


# ==================== 训练器 ====================

class GateTrainer:
    """
    门控网络训练器
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, device='cuda', checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.history = {'train': [], 'val': [], 'test': {}}

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            h_cat = batch['h_cat'].to(self.device)
            h_num = batch['h_num'].to(self.device)
            h_graph = batch['h_graph'].to(self.device)
            p_cat = batch['p_cat'].to(self.device)
            p_num = batch['p_num'].to(self.device)
            p_graph = batch['p_graph'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # 前向传播 - 使用预测概率加权聚合
            p_final, weights = self.model(h_cat, h_num, h_graph, p_cat, p_num, p_graph)

            # 计算损失
            loss = self.criterion(p_final, labels)
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = (p_final > 0.5).float()
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

        return {'loss': avg_loss, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}

        # 收集权重统计
        all_weights = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                h_cat = batch['h_cat'].to(self.device)
                h_num = batch['h_num'].to(self.device)
                h_graph = batch['h_graph'].to(self.device)
                p_cat = batch['p_cat'].to(self.device)
                p_num = batch['p_num'].to(self.device)
                p_graph = batch['p_graph'].to(self.device)
                labels = batch['label'].to(self.device)

                p_final, weights = self.model(h_cat, h_num, h_graph, p_cat, p_num, p_graph)
                loss = self.criterion(p_final, labels)

                total_loss += loss.item()
                predictions = (p_final > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                all_weights.append(weights.cpu())

                _, _, f1_running = compute_binary_f1(counts)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / total:.4f}',
                    'f1': f'{f1_running:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 计算平均权重
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)
        print(f"  平均专家权重: cat={avg_weights[0]:.4f}, num={avg_weights[1]:.4f}, graph={avg_weights[2]:.4f}")

        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best', epoch, {'val_loss': avg_loss, 'val_f1': f1})
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_loss:.4f}, Val F1: {f1:.4f})")

        return {'loss': avg_loss, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def test(self):
        """测试"""
        print("\n开始测试...")
        self.load_checkpoint('best')
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        all_weights = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                h_cat = batch['h_cat'].to(self.device)
                h_num = batch['h_num'].to(self.device)
                h_graph = batch['h_graph'].to(self.device)
                p_cat = batch['p_cat'].to(self.device)
                p_num = batch['p_num'].to(self.device)
                p_graph = batch['p_graph'].to(self.device)
                labels = batch['label'].to(self.device)

                p_final, weights = self.model(h_cat, h_num, h_graph, p_cat, p_num, p_graph)
                loss = self.criterion(p_final, labels)

                total_loss += loss.item()
                predictions = (p_final > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                update_binary_counts(predictions, labels, counts)
                all_weights.append(weights.cpu())

        avg_loss = total_loss / len(self.test_loader)
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)

        # 计算平均权重
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)

        return {
            'loss': avg_loss,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_weights': {
                'cat': avg_weights[0].item(),
                'num': avg_weights[1].item(),
                'graph': avg_weights[2].item()
            }
        }

    def train(self, num_epochs):
        """完整训练流程"""
        print(f"\n{'=' * 60}")
        print("开始训练 MoE Gate Network")
        print(f"{'=' * 60}\n")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 测试
        test_metrics = self.test()
        self.history['test'] = test_metrics

        print(f"\n{'=' * 60}")
        print("MoE Gate Network 测试结果:")
        print(f"{'=' * 60}")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['acc']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"\n  专家权重分布:")
        print(f"    α_cat:   {test_metrics['avg_weights']['cat']:.4f}")
        print(f"    α_num:   {test_metrics['avg_weights']['num']:.4f}")
        print(f"    α_graph: {test_metrics['avg_weights']['graph']:.4f}")
        print(f"{'=' * 60}\n")

        return self.history

    def save_checkpoint(self, name, epoch=None, extra_info=None):
        """保存检查点"""
        path = self.checkpoint_dir / f"gate_{name}.pt"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if extra_info is not None:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)

    def load_checkpoint(self, name):
        """加载检查点"""
        path = self.checkpoint_dir / f"gate_{name}.pt"
        if not path.exists():
            print(f"警告: 检查点 {path} 不存在")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ==================== 主函数 ====================

def main():
    """
    训练门控网络的主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train MoE Gate Network')
    parser.add_argument('--embedding_dir', type=str, default='../../autodl-fs/labeled_embedding',
                        help='专家 embedding 文件目录')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/checkpoints',
                        help='模型检查点保存目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout 比例')
    parser.add_argument('--device', type=str, default='cuda', help='设备')

    args = parser.parse_args()

    # 设备
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载专家 embedding 数据...")
    train_dataset = ExpertEmbeddingDataset('train', args.embedding_dir)
    val_dataset = ExpertEmbeddingDataset('val', args.embedding_dir)
    test_dataset = ExpertEmbeddingDataset('test', args.embedding_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = MoEEnsemble(
        expert_dim=64,
        num_experts=3,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # 训练器
    trainer = GateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # 训练
    history = trainer.train(args.epochs)

    return history


if __name__ == '__main__':
    main()
