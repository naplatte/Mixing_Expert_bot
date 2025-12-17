"""
Expert Importance Gating Network
用于聚合多个专家模型（cat、num、graph、des、post）的预测结果

参考 BotBuster 论文的方法:
1. 将各专家的 64 维表示向量拼接
2. 通过两层 MLP + tanh 激活
3. 最后 softmax 输出归一化权重
4. 加权聚合各专家的预测概率

扩展功能:
- 支持 5 个专家：cat, num, graph, des, post
- des 和 post 专家支持 mask 机制，对于缺失数据的样本自动跳过
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

    输入: 五个专家（cat, num, graph, des, post）的 64 维表示向量拼接 -> 320 维
    输出: 5 维权重向量（softmax 归一化，和为 1）

    网络结构（参考 BotBuster）:
        - 两层 MLP
        - tanh 激活函数
        - softmax 输出层（支持 mask 机制）
    """

    def __init__(self, expert_dim=64, num_experts=5, hidden_dim=64, dropout=0.2):
        """
        Args:
            expert_dim: 每个专家的表示维度（默认 64）
            num_experts: 专家数量（默认 5: cat, num, graph, des, post）
            hidden_dim: 隐藏层维度
            dropout: dropout 比例
        """
        super(MoEGate, self).__init__()

        self.expert_dim = expert_dim
        self.num_experts = num_experts
        input_dim = expert_dim * num_experts  # 64 * 5 = 320

        # 两层 MLP + tanh 激活
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
            # softmax 在 forward 中单独处理
        )

        # 专家名称（用于输出）
        self.expert_names = ['cat', 'num', 'graph', 'des', 'post']

    def forward(self, h_cat, h_num, h_graph, h_des, h_post, expert_mask=None):
        """
        Args:
            h_cat: [batch_size, 64] - Cat Expert 的表示向量
            h_num: [batch_size, 64] - Num Expert 的表示向量
            h_graph: [batch_size, 64] - Graph Expert 的表示向量
            h_des: [batch_size, 64] - Des Expert 的表示向量（缺失数据用零向量）
            h_post: [batch_size, 64] - Post Expert 的表示向量（缺失数据用零向量）
            expert_mask: [batch_size, 5] - 专家有效性掩码（1=有效，0=缺失）
                         如果为 None，则假设所有专家都有效

        Returns:
            weights: [batch_size, 5] - 各专家的权重（masked softmax 归一化）
                     weights[:, 0] -> α_cat
                     weights[:, 1] -> α_num
                     weights[:, 2] -> α_graph
                     weights[:, 3] -> α_des
                     weights[:, 4] -> α_post
        """
        # 拼接五个专家的表示向量
        x = torch.cat([h_cat, h_num, h_graph, h_des, h_post], dim=-1)  # [batch_size, 320]

        # 门控网络
        logits = self.gate(x)  # [batch_size, 5]

        # Masked softmax 归一化
        if expert_mask is not None:
            # 将缺失专家的 logits 设为负无穷，softmax 后权重为 0
            logits = logits.masked_fill(~expert_mask.bool(), float('-inf'))

        weights = F.softmax(logits, dim=-1)  # [batch_size, 5]

        # 处理全部 mask 的情况（理论上不应该发生，因为 cat/num/graph 总是有效）
        # 但为了数值稳定性，将 NaN 替换为均匀分布
        weights = torch.nan_to_num(weights, nan=0.0)

        return weights

    def get_expert_weights(self, h_cat, h_num, h_graph, h_des, h_post, expert_mask=None):
        """
        获取各专家的权重（带名称）

        Returns:
            dict: {'cat': α_cat, 'num': α_num, 'graph': α_graph, 'des': α_des, 'post': α_post}
        """
        weights = self.forward(h_cat, h_num, h_graph, h_des, h_post, expert_mask)
        return {
            'cat': weights[:, 0],
            'num': weights[:, 1],
            'graph': weights[:, 2],
            'des': weights[:, 3],
            'post': weights[:, 4]
        }


class MoEEnsemble(nn.Module):
    """
    完整的 MoE 集成模型（扩展版，支持 5 个专家 + mask 机制 + 负载均衡）

    门控网络输出权重，直接加权聚合各专家的预测概率:
        p_final = α_cat * p_cat + α_num * p_num + α_graph * p_graph + α_des * p_des + α_post * p_post

    对于缺失 des/post 数据的样本，通过 mask 机制自动将对应专家权重置零
    
    新增负载均衡损失，防止门控网络坍塌到单一专家
    """

    def __init__(self, expert_dim=64, num_experts=5, hidden_dim=64, dropout=0.2):
        super(MoEEnsemble, self).__init__()

        self.expert_dim = expert_dim
        self.num_experts = num_experts

        # 门控网络：输入专家表示，输出权重
        self.gate = MoEGate(expert_dim, num_experts, hidden_dim, dropout)

    def forward(self, h_cat, h_num, h_graph, h_des, h_post,
                p_cat, p_num, p_graph, p_des, p_post, expert_mask=None):
        """
        用权重加权各专家的预测概率（支持 mask 机制）

        Args:
            h_cat, h_num, h_graph, h_des, h_post: [batch_size, 64] - 各专家的表示向量
            p_cat, p_num, p_graph, p_des, p_post: [batch_size, 1] - 各专家的预测概率
            expert_mask: [batch_size, 5] - 专家有效性掩码
                         顺序: [cat, num, graph, des, post]
                         1 = 有效，0 = 缺失（该专家不参与加权）
                         如果为 None，则假设所有专家都有效

        Returns:
            p_final: [batch_size, 1] - 最终预测概率
            weights: [batch_size, 5] - 各专家权重 [α_cat, α_num, α_graph, α_des, α_post]
        """
        # 获取门控权重（基于专家表示向量，考虑 mask）
        weights = self.gate(h_cat, h_num, h_graph, h_des, h_post, expert_mask)  # [batch_size, 5]

        # 加权聚合各专家的预测概率
        p_final = (weights[:, 0:1] * p_cat +
                   weights[:, 1:2] * p_num +
                   weights[:, 2:3] * p_graph +
                   weights[:, 3:4] * p_des +
                   weights[:, 4:5] * p_post)  # [batch_size, 1]

        return p_final, weights
    
    def compute_load_balancing_loss(self, weights, expert_mask=None):
        """
        计算负载均衡损失，鼓励门控网络更均匀地使用各专家
        
        参考 Switch Transformer 的负载均衡损失:
        L_balance = num_experts * sum_i(f_i * P_i)
        其中 f_i 是专家 i 被选中的比例，P_i 是专家 i 的平均路由概率
        
        Args:
            weights: [batch_size, num_experts] - 门控权重
            expert_mask: [batch_size, num_experts] - 专家有效性掩码
            
        Returns:
            load_balance_loss: 标量损失值
        """
        batch_size = weights.size(0)
        
        if expert_mask is not None:
            # 只考虑有效的专家
            # 计算每个专家的有效样本数
            valid_counts = expert_mask.sum(dim=0)  # [num_experts]
            # 计算每个专家的平均权重（只在有效样本上）
            masked_weights = weights * expert_mask.float()
            avg_weights = masked_weights.sum(dim=0) / valid_counts.clamp(min=1)  # [num_experts]
        else:
            avg_weights = weights.mean(dim=0)  # [num_experts]
        
        # 计算负载均衡损失：鼓励权重分布更均匀
        # 使用方差作为不均衡度量
        # 理想情况下，每个专家权重应该接近 1/num_valid_experts
        num_valid = expert_mask.sum(dim=1).float().mean() if expert_mask is not None else self.num_experts
        target_weight = 1.0 / num_valid
        
        # 方差损失：权重偏离均匀分布的程度
        load_balance_loss = ((avg_weights - target_weight) ** 2).sum() * self.num_experts
        
        return load_balance_loss


# ==================== 数据集 ====================

class ExpertEmbeddingDataset(Dataset):
    """
    加载各专家的预训练 embedding 和预测概率数据集（支持 5 个专家 + mask）

    从 embedding_dir 目录加载:
        - cat_embeddings.pt (包含 embeddings, probs, labels)
        - num_embeddings.pt
        - graph_embeddings.pt
        - des_embeddings.pt (可选，包含 mask 标记缺失数据)
        - post_embeddings.pt (可选，包含 mask 标记缺失数据)
    """

    def __init__(self, split='train', embedding_dir='../../autodl-fs/labeled_embedding', expert_dim=64):
        """
        Args:
            split: 'train', 'val', 或 'test'
            embedding_dir: embedding 文件所在目录
            expert_dim: 专家表示向量维度（默认 64）
        """
        self.split = split
        self.expert_dim = expert_dim
        embedding_path = Path(embedding_dir)

        # 加载基础三个专家的 embedding（这三个专家所有样本都有数据）
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
        self.num_samples = len(self.labels)

        # 验证数据一致性
        assert len(self.h_cat) == len(self.h_num) == len(self.h_graph) == self.num_samples, \
            f"数据长度不一致: cat={len(self.h_cat)}, num={len(self.h_num)}, graph={len(self.h_graph)}, labels={self.num_samples}"

        # ========== 加载 des 专家数据（可选） ==========
        des_path = embedding_path / 'des_embeddings.pt'
        if des_path.exists():
            des_data = torch.load(des_path, map_location='cpu')
            self.h_des = des_data[split]['embeddings']  # [N, 64]
            self.p_des = des_data[split]['probs']  # [N, 1]
            self.des_mask = des_data[split]['mask']  # [N] bool tensor
            des_valid = self.des_mask.sum().item()
            print(f"  [des] 有效样本: {des_valid}/{self.num_samples}")
        else:
            # 如果没有 des 数据，用零向量填充，mask 全为 False
            print(f"  [des] 未找到 {des_path}，使用零向量填充")
            self.h_des = torch.zeros(self.num_samples, expert_dim)
            self.p_des = torch.zeros(self.num_samples, 1)
            self.des_mask = torch.zeros(self.num_samples, dtype=torch.bool)

        # ========== 加载 post 专家数据（可选） ==========
        post_path = embedding_path / 'post_embeddings.pt'
        if post_path.exists():
            post_data = torch.load(post_path, map_location='cpu')
            self.h_post = post_data[split]['embeddings']  # [N, 64]
            self.p_post = post_data[split]['probs']  # [N, 1]
            self.post_mask = post_data[split]['mask']  # [N] bool tensor
            post_valid = self.post_mask.sum().item()
            print(f"  [post] 有效样本: {post_valid}/{self.num_samples}")
        else:
            # 如果没有 post 数据，用零向量填充，mask 全为 False
            print(f"  [post] 未找到 {post_path}，使用零向量填充")
            self.h_post = torch.zeros(self.num_samples, expert_dim)
            self.p_post = torch.zeros(self.num_samples, 1)
            self.post_mask = torch.zeros(self.num_samples, dtype=torch.bool)

        print(f"[{split.upper()}] 加载 {self.num_samples} 个样本")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 构建专家 mask: [cat, num, graph, des, post]
        # cat, num, graph 总是有效 (1)，des 和 post 根据数据情况
        expert_mask = torch.tensor([
            True,  # cat 总是有效
            True,  # num 总是有效
            True,  # graph 总是有效
            self.des_mask[idx].item(),   # des 根据数据
            self.post_mask[idx].item()   # post 根据数据
        ], dtype=torch.float32)

        return {
            'h_cat': self.h_cat[idx],
            'h_num': self.h_num[idx],
            'h_graph': self.h_graph[idx],
            'h_des': self.h_des[idx],
            'h_post': self.h_post[idx],
            'p_cat': self.p_cat[idx],
            'p_num': self.p_num[idx],
            'p_graph': self.p_graph[idx],
            'p_des': self.p_des[idx],
            'p_post': self.p_post[idx],
            'expert_mask': expert_mask,  # [5] - 专家有效性掩码
            'label': self.labels[idx]
        }


# ==================== 训练器 ====================

class GateTrainer:
    """
    门控网络训练器
    
    新增负载均衡损失系数 load_balance_weight，防止门控网络坍塌
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, device='cuda', checkpoint_dir='checkpoints',
                 load_balance_weight=0.1):
        """
        Args:
            load_balance_weight: 负载均衡损失的权重系数（默认0.1）
                                 设为0则不使用负载均衡损失
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.load_balance_weight = load_balance_weight

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
            # 加载 5 个专家的数据
            h_cat = batch['h_cat'].to(self.device)
            h_num = batch['h_num'].to(self.device)
            h_graph = batch['h_graph'].to(self.device)
            h_des = batch['h_des'].to(self.device)
            h_post = batch['h_post'].to(self.device)
            p_cat = batch['p_cat'].to(self.device)
            p_num = batch['p_num'].to(self.device)
            p_graph = batch['p_graph'].to(self.device)
            p_des = batch['p_des'].to(self.device)
            p_post = batch['p_post'].to(self.device)
            expert_mask = batch['expert_mask'].to(self.device)  # [batch, 5]
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # 前向传播 - 使用预测概率加权聚合（带 mask）
            p_final, weights = self.model(
                h_cat, h_num, h_graph, h_des, h_post,
                p_cat, p_num, p_graph, p_des, p_post,
                expert_mask
            )

            # 计算损失 = BCE损失 + 负载均衡损失
            bce_loss = self.criterion(p_final, labels)
            
            if self.load_balance_weight > 0:
                lb_loss = self.model.compute_load_balancing_loss(weights, expert_mask)
                loss = bce_loss + self.load_balance_weight * lb_loss
            else:
                loss = bce_loss
                
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
                # 加载 5 个专家的数据
                h_cat = batch['h_cat'].to(self.device)
                h_num = batch['h_num'].to(self.device)
                h_graph = batch['h_graph'].to(self.device)
                h_des = batch['h_des'].to(self.device)
                h_post = batch['h_post'].to(self.device)
                p_cat = batch['p_cat'].to(self.device)
                p_num = batch['p_num'].to(self.device)
                p_graph = batch['p_graph'].to(self.device)
                p_des = batch['p_des'].to(self.device)
                p_post = batch['p_post'].to(self.device)
                expert_mask = batch['expert_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                p_final, weights = self.model(
                    h_cat, h_num, h_graph, h_des, h_post,
                    p_cat, p_num, p_graph, p_des, p_post,
                    expert_mask
                )
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

        # 计算平均权重（5 个专家）
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)
        print(f"  平均专家权重: cat={avg_weights[0]:.4f}, num={avg_weights[1]:.4f}, graph={avg_weights[2]:.4f}, des={avg_weights[3]:.4f}, post={avg_weights[4]:.4f}")

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
                # 加载 5 个专家的数据
                h_cat = batch['h_cat'].to(self.device)
                h_num = batch['h_num'].to(self.device)
                h_graph = batch['h_graph'].to(self.device)
                h_des = batch['h_des'].to(self.device)
                h_post = batch['h_post'].to(self.device)
                p_cat = batch['p_cat'].to(self.device)
                p_num = batch['p_num'].to(self.device)
                p_graph = batch['p_graph'].to(self.device)
                p_des = batch['p_des'].to(self.device)
                p_post = batch['p_post'].to(self.device)
                expert_mask = batch['expert_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                p_final, weights = self.model(
                    h_cat, h_num, h_graph, h_des, h_post,
                    p_cat, p_num, p_graph, p_des, p_post,
                    expert_mask
                )
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

        # 计算平均权重（5 个专家）
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
                'graph': avg_weights[2].item(),
                'des': avg_weights[3].item(),
                'post': avg_weights[4].item()
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
        print(f"    α_des:   {test_metrics['avg_weights']['des']:.4f}")
        print(f"    α_post:  {test_metrics['avg_weights']['post']:.4f}")
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
    parser.add_argument('--load_balance_weight', type=float, default=0.1,
                        help='负载均衡损失权重（防止门控坍塌，默认0.1，设为0禁用）')

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

    # 创建模型（5 个专家）
    model = MoEEnsemble(
        expert_dim=64,
        num_experts=5,  # cat, num, graph, des, post
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # 训练器（带负载均衡损失）
    trainer = GateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        load_balance_weight=args.load_balance_weight
    )

    # 训练
    history = trainer.train(args.epochs)

    return history


if __name__ == '__main__':
    main()
