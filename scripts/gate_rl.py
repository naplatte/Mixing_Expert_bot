"""
Reinforcement Learning based Gating Network
使用强化学习（REINFORCE算法）训练门控网络

核心思想：
- 状态(State): 各专家的embedding表示 [h_cat, h_num, h_graph, h_des, h_post]
- 动作(Action): 选择专家权重分布（从策略网络采样）
- 奖励(Reward): 基于预测正确性的奖励
  - 预测正确: +1
  - 预测错误: -1
  - 可选: 加入多样性奖励，鼓励使用多个专家

优势：
1. 不会像监督学习那样容易坍塌到单一专家
2. 可以通过奖励函数设计来鼓励专家多样性
3. 更灵活的优化目标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import update_binary_counts, compute_binary_f1


class RLGatePolicy(nn.Module):
    """
    强化学习策略网络（门控网络）
    
    输入: 五个专家的embedding表示拼接 -> 320维
    输出: 专家选择的概率分布（用于采样动作）
    
    支持两种动作空间：
    1. 离散动作: 选择单个专家 (Categorical)
    2. 连续动作: 输出权重分布 (Dirichlet)
    """
    
    def __init__(self, expert_dim=64, num_experts=5, hidden_dim=128, 
                 action_type='continuous', dropout=0.2):
        """
        Args:
            expert_dim: 每个专家的表示维度
            num_experts: 专家数量
            hidden_dim: 隐藏层维度
            action_type: 'discrete'(选单个专家) 或 'continuous'(输出权重分布)
            dropout: dropout比例
        """
        super(RLGatePolicy, self).__init__()
        
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.action_type = action_type
        input_dim = expert_dim * num_experts  # 64 * 5 = 320
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        
        if action_type == 'discrete':
            # 离散动作: 输出每个专家被选中的logits
            self.action_head = nn.Linear(hidden_dim // 2, num_experts)
        else:
            # 连续动作: 输出Dirichlet分布的concentration参数
            # 使用softplus确保参数为正
            self.action_head = nn.Linear(hidden_dim // 2, num_experts)
        
        # 价值网络（用于baseline，减少方差）
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.expert_names = ['cat', 'num', 'graph', 'des', 'post']
    
    def forward(self, h_cat, h_num, h_graph, h_des, h_post, expert_mask=None):
        """
        前向传播，返回动作分布和状态价值
        
        Args:
            h_cat, h_num, h_graph, h_des, h_post: [batch_size, 64] 各专家embedding
            expert_mask: [batch_size, 5] 专家有效性掩码
            
        Returns:
            action_dist: 动作分布（用于采样）
            state_value: 状态价值估计
        """
        # 拼接专家表示
        x = torch.cat([h_cat, h_num, h_graph, h_des, h_post], dim=-1)  # [batch, 320]
        
        # 策略网络
        policy_features = self.policy_net(x)
        action_params = self.action_head(policy_features)
        
        # 状态价值
        state_value = self.value_net(x)
        
        if self.action_type == 'discrete':
            # 离散动作空间
            if expert_mask is not None:
                # 将无效专家的logits设为负无穷
                action_params = action_params.masked_fill(~expert_mask.bool(), float('-inf'))
            action_dist = Categorical(logits=action_params)
        else:
            # 连续动作空间 (Dirichlet分布)
            # concentration参数需要为正，使用softplus + 小常数
            concentration = F.softplus(action_params) + 0.1  # [batch, 5]
            if expert_mask is not None:
                # 将无效专家的concentration设为很小的值
                concentration = concentration * expert_mask.float() + 0.01 * (~expert_mask.bool()).float()
            action_dist = Dirichlet(concentration)
        
        return action_dist, state_value
    
    def get_action(self, h_cat, h_num, h_graph, h_des, h_post, expert_mask=None, 
                   deterministic=False):
        """
        获取动作（采样或确定性）
        
        Returns:
            weights: [batch_size, num_experts] 专家权重
            log_prob: [batch_size] 动作的对数概率
            state_value: [batch_size, 1] 状态价值
        """
        action_dist, state_value = self.forward(
            h_cat, h_num, h_graph, h_des, h_post, expert_mask
        )
        
        if deterministic:
            if self.action_type == 'discrete':
                action = action_dist.probs.argmax(dim=-1)
                weights = F.one_hot(action, self.num_experts).float()
            else:
                # Dirichlet的众数
                weights = action_dist.concentration / action_dist.concentration.sum(dim=-1, keepdim=True)
            log_prob = action_dist.log_prob(weights if self.action_type == 'continuous' else action)
        else:
            if self.action_type == 'discrete':
                action = action_dist.sample()
                weights = F.one_hot(action, self.num_experts).float()
                log_prob = action_dist.log_prob(action)
            else:
                weights = action_dist.sample()  # [batch, 5]
                log_prob = action_dist.log_prob(weights)
        
        return weights, log_prob, state_value


class RLGateEnsemble(nn.Module):
    """
    RL门控的MoE集成模型
    
    使用策略网络输出的权重加权聚合各专家预测
    """
    
    def __init__(self, expert_dim=64, num_experts=5, hidden_dim=128,
                 action_type='continuous', dropout=0.2):
        super(RLGateEnsemble, self).__init__()
        
        self.policy = RLGatePolicy(
            expert_dim=expert_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            action_type=action_type,
            dropout=dropout
        )
        self.num_experts = num_experts
    
    def forward(self, h_cat, h_num, h_graph, h_des, h_post,
                p_cat, p_num, p_graph, p_des, p_post, 
                expert_mask=None, deterministic=False):
        """
        前向传播
        
        Returns:
            p_final: [batch, 1] 最终预测概率
            weights: [batch, 5] 专家权重
            log_prob: [batch] 动作对数概率
            state_value: [batch, 1] 状态价值
        """
        # 获取权重
        weights, log_prob, state_value = self.policy.get_action(
            h_cat, h_num, h_graph, h_des, h_post, 
            expert_mask, deterministic
        )
        
        # 加权聚合预测概率
        p_final = (weights[:, 0:1] * p_cat +
                   weights[:, 1:2] * p_num +
                   weights[:, 2:3] * p_graph +
                   weights[:, 3:4] * p_des +
                   weights[:, 4:5] * p_post)
        
        return p_final, weights, log_prob, state_value


# ==================== 数据集（复用gate.py的） ====================

class ExpertEmbeddingDataset(Dataset):
    """加载各专家的预训练embedding和预测概率"""
    
    def __init__(self, split='train', embedding_dir='../../autodl-fs/labeled_embedding', expert_dim=64):
        self.split = split
        self.expert_dim = expert_dim
        embedding_path = Path(embedding_dir)
        
        # 加载基础三个专家
        cat_data = torch.load(embedding_path / 'cat_embeddings.pt', map_location='cpu')
        num_data = torch.load(embedding_path / 'num_embeddings.pt', map_location='cpu')
        graph_data = torch.load(embedding_path / 'graph_embeddings.pt', map_location='cpu')
        
        self.h_cat = cat_data[split]['embeddings']
        self.h_num = num_data[split]['embeddings']
        self.h_graph = graph_data[split]['embeddings']
        self.p_cat = cat_data[split]['probs']
        self.p_num = num_data[split]['probs']
        self.p_graph = graph_data[split]['probs']
        self.labels = cat_data[split]['labels']
        self.num_samples = len(self.labels)
        
        # 加载des专家（可选）
        des_path = embedding_path / 'des_embeddings.pt'
        if des_path.exists():
            des_data = torch.load(des_path, map_location='cpu')
            self.h_des = des_data[split]['embeddings']
            self.p_des = des_data[split]['probs']
            self.des_mask = des_data[split].get('mask', torch.ones(self.num_samples, dtype=torch.bool))
        else:
            print(f"  [des] 未找到 {des_path}，使用零向量填充")
            self.h_des = torch.zeros(self.num_samples, expert_dim)
            self.p_des = torch.zeros(self.num_samples, 1)
            self.des_mask = torch.zeros(self.num_samples, dtype=torch.bool)
        
        # 加载post专家（可选）
        post_path = embedding_path / 'post_embeddings.pt'
        if post_path.exists():
            post_data = torch.load(post_path, map_location='cpu')
            self.h_post = post_data[split]['embeddings']
            self.p_post = post_data[split]['probs']
            self.post_mask = post_data[split].get('mask', torch.ones(self.num_samples, dtype=torch.bool))
            valid_count = self.post_mask.sum().item()
            print(f"  [post] 有效样本: {valid_count}/{self.num_samples}")
        else:
            print(f"  [post] 未找到 {post_path}，使用零向量填充")
            self.h_post = torch.zeros(self.num_samples, expert_dim)
            self.p_post = torch.zeros(self.num_samples, 1)
            self.post_mask = torch.zeros(self.num_samples, dtype=torch.bool)
        
        print(f"[{split.upper()}] 加载 {self.num_samples} 个样本")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        expert_mask = torch.tensor([
            True, True, True,
            self.des_mask[idx].item(),
            self.post_mask[idx].item()
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
            'expert_mask': expert_mask,
            'label': self.labels[idx]
        }


# ==================== RL训练器 ====================

class RLGateTrainer:
    """
    REINFORCE算法训练门控网络
    
    奖励设计：
    1. 基础奖励: 预测正确+1，错误-1
    2. 多样性奖励: 鼓励使用多个专家（熵奖励）
    3. 可选: 基于F1的奖励
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, device='cuda', checkpoint_dir='checkpoints',
                 gamma=0.99, entropy_coef=0.01, value_coef=0.5,
                 diversity_coef=0.1):
        """
        Args:
            gamma: 折扣因子
            entropy_coef: 熵正则化系数（鼓励探索）
            value_coef: 价值损失系数
            diversity_coef: 多样性奖励系数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.diversity_coef = diversity_coef
        
        self.best_val_f1 = 0.0
        self.history = {'train': [], 'val': [], 'test': {}}
    
    def compute_reward(self, predictions, labels, weights):
        """
        计算奖励
        
        Args:
            predictions: [batch] 预测结果 (0或1)
            labels: [batch] 真实标签
            weights: [batch, 5] 专家权重
            
        Returns:
            rewards: [batch] 每个样本的奖励
        """
        batch_size = predictions.size(0)
        
        # 基础奖励: 预测正确+1，错误-1
        correct = (predictions == labels.squeeze()).float()
        base_reward = 2 * correct - 1  # 正确+1，错误-1
        
        # 多样性奖励: 权重分布的熵（越均匀熵越大）
        # 熵 = -sum(p * log(p))，最大值为log(num_experts)
        weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        max_entropy = np.log(self.model.num_experts)
        normalized_entropy = weight_entropy / max_entropy  # 归一化到[0,1]
        
        # 总奖励
        rewards = base_reward + self.diversity_coef * normalized_entropy
        
        return rewards
    
    def train_epoch(self, epoch):
        """训练一个epoch（REINFORCE算法）"""
        self.model.train()
        
        total_reward = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        all_weights = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # 加载数据
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
            
            self.optimizer.zero_grad()
            
            # 前向传播（采样动作）
            p_final, weights, log_prob, state_value = self.model(
                h_cat, h_num, h_graph, h_des, h_post,
                p_cat, p_num, p_graph, p_des, p_post,
                expert_mask, deterministic=False
            )
            
            # 计算预测和奖励
            predictions = (p_final > 0.5).float().squeeze()
            rewards = self.compute_reward(predictions, labels, weights)
            
            # 计算优势 (Advantage = Reward - Value)
            advantage = rewards - state_value.squeeze()
            
            # 策略损失 (REINFORCE with baseline)
            policy_loss = -(log_prob * advantage.detach()).mean()
            
            # 价值损失 (MSE)
            value_loss = F.mse_loss(state_value.squeeze(), rewards.detach())
            
            # 熵正则化（鼓励探索）
            # 对于Dirichlet分布，使用权重的熵
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_reward += rewards.sum().item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            correct += (predictions == labels.squeeze()).sum().item()
            total += labels.size(0)
            
            update_binary_counts(predictions.unsqueeze(1), labels, counts)
            all_weights.append(weights.detach().cpu())
            
            _, _, f1_running = compute_binary_f1(counts)
            pbar.set_postfix({
                'reward': f'{rewards.mean().item():.3f}',
                'acc': f'{correct/total:.4f}',
                'f1': f'{f1_running:.4f}'
            })
        
        # 计算epoch统计
        avg_reward = total_reward / total
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)
        
        return {
            'reward': avg_reward,
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'policy_loss': total_policy_loss / len(self.train_loader),
            'value_loss': total_value_loss / len(self.train_loader),
            'entropy': total_entropy / len(self.train_loader),
            'avg_weights': avg_weights.numpy()
        }
    
    def validate(self, epoch):
        """验证（确定性策略）"""
        self.model.eval()
        
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        all_weights = []
        total_reward = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
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
                
                # 确定性策略
                p_final, weights, _, _ = self.model(
                    h_cat, h_num, h_graph, h_des, h_post,
                    p_cat, p_num, p_graph, p_des, p_post,
                    expert_mask, deterministic=True
                )
                
                predictions = (p_final > 0.5).float().squeeze()
                rewards = self.compute_reward(predictions, labels, weights)
                
                total_reward += rewards.sum().item()
                correct += (predictions == labels.squeeze()).sum().item()
                total += labels.size(0)
                
                update_binary_counts(predictions.unsqueeze(1), labels, counts)
                all_weights.append(weights.cpu())
        
        avg_reward = total_reward / total
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)
        
        print(f"  平均专家权重: cat={avg_weights[0]:.4f}, num={avg_weights[1]:.4f}, "
              f"graph={avg_weights[2]:.4f}, des={avg_weights[3]:.4f}, post={avg_weights[4]:.4f}")
        
        # 保存最佳模型
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.save_checkpoint('best', epoch, {'val_f1': f1})
            print(f"  ✓ 保存最佳模型 (Val F1: {f1:.4f})")
        
        return {
            'reward': avg_reward,
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'avg_weights': avg_weights.numpy()
        }
    
    def test(self):
        """测试"""
        print("\n开始测试...")
        self.load_checkpoint('best')
        self.model.eval()
        
        correct = 0
        total = 0
        counts = {'tp': 0, 'fp': 0, 'fn': 0}
        all_weights = []
        total_reward = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
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
                
                p_final, weights, _, _ = self.model(
                    h_cat, h_num, h_graph, h_des, h_post,
                    p_cat, p_num, p_graph, p_des, p_post,
                    expert_mask, deterministic=True
                )
                
                predictions = (p_final > 0.5).float().squeeze()
                rewards = self.compute_reward(predictions, labels, weights)
                
                total_reward += rewards.sum().item()
                correct += (predictions == labels.squeeze()).sum().item()
                total += labels.size(0)
                
                update_binary_counts(predictions.unsqueeze(1), labels, counts)
                all_weights.append(weights.cpu())
        
        avg_reward = total_reward / total
        acc = correct / total
        precision, recall, f1 = compute_binary_f1(counts)
        
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = all_weights.mean(dim=0)
        
        return {
            'reward': avg_reward,
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
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
        print("开始训练 RL Gate Network (REINFORCE)")
        print(f"{'=' * 60}")
        print(f"  熵系数: {self.entropy_coef}")
        print(f"  多样性系数: {self.diversity_coef}")
        print(f"  价值系数: {self.value_coef}")
        print()
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Reward: {train_metrics['reward']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Reward: {val_metrics['reward']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"  Train Weights: {train_metrics['avg_weights']}")
        
        # 测试
        test_metrics = self.test()
        self.history['test'] = test_metrics
        
        print(f"\n{'=' * 60}")
        print("RL Gate Network 测试结果:")
        print(f"{'=' * 60}")
        print(f"  Reward: {test_metrics['reward']:.4f}")
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
        path = self.checkpoint_dir / f"gate_rl_{name}.pt"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if extra_info is not None:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, name):
        """加载检查点"""
        path = self.checkpoint_dir / f"gate_rl_{name}.pt"
        if not path.exists():
            print(f"警告: 检查点 {path} 不存在")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ==================== 主函数 ====================

def main():
    """训练RL门控网络"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Gate Network')
    parser.add_argument('--embedding_dir', type=str, default='../../autodl-fs/labeled_embedding',
                        help='专家embedding文件目录')
    parser.add_argument('--checkpoint_dir', type=str, default='../../autodl-fs/checkpoints',
                        help='模型检查点保存目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比例')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    # RL特有参数
    parser.add_argument('--action_type', type=str, default='continuous',
                        choices=['discrete', 'continuous'],
                        help='动作类型: discrete(选单个专家) 或 continuous(权重分布)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='熵正则化系数（鼓励探索）')
    parser.add_argument('--diversity_coef', type=float, default=0.1,
                        help='多样性奖励系数（鼓励使用多个专家）')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='价值损失系数')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    
    args = parser.parse_args()
    
    # 设备
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载专家embedding数据...")
    train_dataset = ExpertEmbeddingDataset('train', args.embedding_dir)
    val_dataset = ExpertEmbeddingDataset('val', args.embedding_dir)
    test_dataset = ExpertEmbeddingDataset('test', args.embedding_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = RLGateEnsemble(
        expert_dim=64,
        num_experts=5,
        hidden_dim=args.hidden_dim,
        action_type=args.action_type,
        dropout=args.dropout
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练器
    trainer = RLGateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        diversity_coef=args.diversity_coef
    )
    
    # 训练
    history = trainer.train(args.epochs)
    
    return history


if __name__ == '__main__':
    main()
